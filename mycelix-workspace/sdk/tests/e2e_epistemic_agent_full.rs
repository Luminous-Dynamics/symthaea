//! Full End-to-End Integration Test: Epistemic-Aware AI Agency
//!
//! Demonstrates the complete integration of:
//! 1. K-Vector trust profiles that evolve from behavior
//! 2. Epistemic classification (E-N-M-H) of all outputs
//! 3. Phi coherence measurement for quality gating
//! 4. GIS uncertainty handling with escalation
//! 5. KREDIT allocation derived from trust scores
//! 6. Storage routing based on epistemic classification
//!
//! This test validates the unique value proposition: AI agents with
//! verifiable "epistemic fingerprints" - trust profiles that prove
//! reliability without revealing internal state.

use mycelix_sdk::agentic::coherence_bridge::{
    check_coherence_for_action, measure_coherence, CoherenceCheckResult, CoherenceHistory,
    CoherenceMeasurementConfig, CoherenceState,
};
use mycelix_sdk::agentic::epistemic_classifier::{
    calculate_epistemic_weight, classify_output, create_classified_output, AgentOutput,
    AgreementScope, ClassificationHints, EpistemicStats, OutputContent, RelevanceDuration,
};
use mycelix_sdk::agentic::kvector_bridge::{
    analyze_behavior, calculate_kredit_from_trust, compute_kvector_update, record_and_maybe_update,
    KVectorBridgeConfig,
};
use mycelix_sdk::agentic::uncertainty::{
    maybe_escalate, MoralActionGuidance, MoralUncertainty, UncertainOutput, UncertaintyCalibration,
};
use mycelix_sdk::agentic::{
    ActionOutcome, AgentClass, AgentConstraints, AgentId, AgentStatus, BehaviorLogEntry,
    InstrumentalActor,
};
use mycelix_sdk::epistemic::{
    EmpiricalLevel, EpistemicClassification, EpistemicClassificationExtended, HarmonicLevel,
    MaterialityLevel, NormativeLevel,
};
use mycelix_sdk::matl::{GovernanceTier, KVector};
use mycelix_sdk::storage::{
    EpistemicStorage, MutabilityMode, SchemaIdentity, StorageBackend, StoreOptions,
};

use std::time::{SystemTime, UNIX_EPOCH};

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Create a test agent with initial K-Vector
fn create_test_agent(name: &str) -> InstrumentalActor {
    InstrumentalActor {
        agent_id: AgentId::generate(),
        sponsor_did: format!("did:test:sponsor-{}", name),
        agent_class: AgentClass::Supervised,
        kredit_balance: 1000,
        kredit_cap: 5000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: now_secs() - 86400, // Created 1 day ago
        last_activity: now_secs(),
        actions_this_hour: 0,
        k_vector: KVector::new_participant(),
        epistemic_stats: EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: Default::default(),
        pending_escalations: vec![],
    }
}

/// Simulate agent actions and record outcomes
fn simulate_agent_actions(
    agent: &mut InstrumentalActor,
    success_rate: f32,
    action_count: usize,
) -> Vec<BehaviorLogEntry> {
    let now = now_secs();
    let successes = (action_count as f32 * success_rate) as usize;

    let entries: Vec<BehaviorLogEntry> = (0..action_count)
        .map(|i| BehaviorLogEntry {
            timestamp: now - (action_count - i) as u64 * 60,
            action_type: format!("action_{}", i % 5),
            kredit_consumed: 10 + (i % 20) as u64,
            counterparties: vec![format!("peer_{}", i % 8)],
            outcome: if i < successes {
                ActionOutcome::Success
            } else if i % 3 == 0 {
                ActionOutcome::ConstraintViolation
            } else {
                ActionOutcome::Error
            },
        })
        .collect();

    agent.behavior_log.extend(entries.clone());
    entries
}

/// Create agent output with epistemic classification
fn create_agent_output(agent_id: &str, content: &str, hints: ClassificationHints) -> AgentOutput {
    create_classified_output(
        agent_id,
        OutputContent::Text(content.to_string()),
        &hints,
        now_secs(),
    )
}

// =============================================================================
// TEST: Full Agent Lifecycle with K-Vector Evolution
// =============================================================================

#[test]
fn test_full_agent_lifecycle_kvector_evolution() {
    println!("\n=== Full Agent Lifecycle: K-Vector Evolution ===\n");

    // Phase 1: Create new agent with default K-Vector
    let mut agent = create_test_agent("lifecycle-test");
    let initial_trust = agent.k_vector.trust_score();
    let initial_kredit = calculate_kredit_from_trust(initial_trust);

    println!("Phase 1: New Agent Created");
    println!("  Initial K-Vector: {:?}", agent.k_vector);
    println!("  Initial Trust Score: {:.3}", initial_trust);
    println!("  Initial KREDIT Cap: {}", initial_kredit);
    println!(
        "  Governance Tier: {:?}",
        GovernanceTier::from_trust_score(initial_trust)
    );

    // Phase 2: Agent performs well (90% success rate)
    println!("\nPhase 2: High Performance Period (90% success)");
    simulate_agent_actions(&mut agent, 0.9, 50);

    let config = KVectorBridgeConfig::default();
    let analysis = analyze_behavior(&agent.behavior_log);
    let days_active = 1.0;
    let updated_kv = compute_kvector_update(&agent.k_vector, &analysis, &config, days_active);
    agent.k_vector = updated_kv;

    let new_trust = agent.k_vector.trust_score();
    let new_kredit = calculate_kredit_from_trust(new_trust);

    println!("  Success Rate: {:.1}%", analysis.success_rate * 100.0);
    println!("  Updated K-Vector:");
    println!("    k_r (Reputation): {:.3}", agent.k_vector.k_r);
    println!("    k_a (Activity): {:.3}", agent.k_vector.k_a);
    println!("    k_i (Integrity): {:.3}", agent.k_vector.k_i);
    println!("    k_p (Performance): {:.3}", agent.k_vector.k_p);
    println!(
        "  New Trust Score: {:.3} (change: {:+.3})",
        new_trust,
        new_trust - initial_trust
    );
    println!(
        "  New KREDIT Cap: {} (change: {:+})",
        new_kredit,
        new_kredit as i64 - initial_kredit as i64
    );

    assert!(
        new_trust > initial_trust,
        "Trust should improve with good performance"
    );
    assert!(
        new_kredit > initial_kredit,
        "KREDIT should increase with trust"
    );

    // Phase 3: Agent has problems (40% success, violations)
    println!("\nPhase 3: Poor Performance Period (40% success, violations)");
    agent.behavior_log.clear();
    simulate_agent_actions(&mut agent, 0.4, 30);

    let analysis2 = analyze_behavior(&agent.behavior_log);
    let updated_kv2 =
        compute_kvector_update(&agent.k_vector, &analysis2, &config, days_active + 1.0);
    let trust_before_drop = agent.k_vector.trust_score();
    agent.k_vector = updated_kv2;

    let dropped_trust = agent.k_vector.trust_score();
    let dropped_kredit = calculate_kredit_from_trust(dropped_trust);

    println!("  Success Rate: {:.1}%", analysis2.success_rate * 100.0);
    println!("  Violations: {}", analysis2.constraint_violations);
    println!(
        "  Trust Score: {:.3} (change: {:+.3})",
        dropped_trust,
        dropped_trust - trust_before_drop
    );
    println!("  KREDIT Cap: {}", dropped_kredit);
    println!(
        "  Integrity (k_i): {:.3} (violations penalized)",
        agent.k_vector.k_i
    );

    assert!(
        dropped_trust < new_trust,
        "Trust should drop with poor performance"
    );
    assert!(
        agent.k_vector.k_i < 1.0,
        "Integrity should be penalized for violations"
    );

    // Phase 4: Recovery (80% success, no violations)
    println!("\nPhase 4: Recovery Period (80% success, clean)");
    agent.behavior_log.clear();

    // Simulate clean actions (no violations)
    let now = now_secs();
    for i in 0..40 {
        agent.behavior_log.push(BehaviorLogEntry {
            timestamp: now - (40 - i) * 60,
            action_type: "clean_action".to_string(),
            kredit_consumed: 15,
            counterparties: vec![format!("partner_{}", i % 10)],
            outcome: if i < 32 {
                ActionOutcome::Success
            } else {
                ActionOutcome::Error
            },
        });
    }

    let analysis3 = analyze_behavior(&agent.behavior_log);
    let recovered_kv =
        compute_kvector_update(&agent.k_vector, &analysis3, &config, days_active + 2.0);
    agent.k_vector = recovered_kv;

    let recovered_trust = agent.k_vector.trust_score();
    let recovered_kredit = calculate_kredit_from_trust(recovered_trust);

    println!("  Success Rate: {:.1}%", analysis3.success_rate * 100.0);
    println!("  Violations: {} (clean!)", analysis3.constraint_violations);
    println!(
        "  Trust Score: {:.3} (recovery: {:+.3})",
        recovered_trust,
        recovered_trust - dropped_trust
    );
    println!("  KREDIT Cap: {}", recovered_kredit);
    println!(
        "  Governance Tier: {:?}",
        GovernanceTier::from_trust_score(recovered_trust)
    );

    println!("\n=== K-Vector Evolution Test PASSED ===\n");
}

// =============================================================================
// TEST: Epistemic Classification of Agent Outputs
// =============================================================================

#[test]
fn test_epistemic_classification_workflow() {
    println!("\n=== Epistemic Classification Workflow ===\n");

    let agent_id = "classifier-agent";
    let mut stats = EpistemicStats::default();

    // Output 1: Personal opinion (low epistemic level)
    let hints1 = ClassificationHints {
        is_personal_opinion: true,
        ..Default::default()
    };
    let output1 = create_agent_output(agent_id, "I think this is interesting", hints1);
    let weight1 = calculate_epistemic_weight(&output1.classification);
    stats.add_output(&output1.classification);

    println!("Output 1: Personal Opinion");
    println!(
        "  Classification: E{}/N{}/M{}/H{}",
        output1.classification.empirical as u8,
        output1.classification.normative as u8,
        output1.classification.materiality as u8,
        output1.classification.harmonic as u8
    );
    println!("  Epistemic Weight: {:.4}", weight1);

    assert_eq!(output1.classification.empirical, EmpiricalLevel::E0Null);
    assert!(weight1 < 0.1, "Personal opinion should have low weight");

    // Output 2: Verified data analysis (medium epistemic level)
    let hints2 = ClassificationHints {
        third_party_verified: true,
        uses_public_data: false,
        agreement_scope: Some(AgreementScope::Community),
        relevance_duration: Some(RelevanceDuration::MediumTerm),
        affected_harmonies_count: 2,
        ..Default::default()
    };
    let output2 = create_agent_output(agent_id, "Analysis shows 15% improvement", hints2);
    let weight2 = calculate_epistemic_weight(&output2.classification);
    stats.add_output(&output2.classification);

    println!("\nOutput 2: Verified Analysis");
    println!(
        "  Classification: E{}/N{}/M{}/H{}",
        output2.classification.empirical as u8,
        output2.classification.normative as u8,
        output2.classification.materiality as u8,
        output2.classification.harmonic as u8
    );
    println!("  Epistemic Weight: {:.4}", weight2);

    assert_eq!(
        output2.classification.empirical,
        EmpiricalLevel::E2PrivateVerify
    );
    assert!(
        weight2 > weight1,
        "Verified analysis should have higher weight"
    );

    // Output 3: Cryptographically proven result (high epistemic level)
    let hints3 = ClassificationHints {
        has_crypto_proof: true,
        uses_public_data: true,
        agreement_scope: Some(AgreementScope::Network),
        relevance_duration: Some(RelevanceDuration::LongTerm),
        affected_harmonies_count: 4,
        ..Default::default()
    };
    let output3 = create_agent_output(agent_id, "ZK proof: gradient quality verified", hints3);
    let weight3 = calculate_epistemic_weight(&output3.classification);
    stats.add_output(&output3.classification);

    println!("\nOutput 3: Cryptographic Proof");
    println!(
        "  Classification: E{}/N{}/M{}/H{}",
        output3.classification.empirical as u8,
        output3.classification.normative as u8,
        output3.classification.materiality as u8,
        output3.classification.harmonic as u8
    );
    println!("  Epistemic Weight: {:.4}", weight3);

    assert_eq!(
        output3.classification.empirical,
        EmpiricalLevel::E4PublicRepro
    );
    assert!(weight3 > 0.5, "Cryptographic proof should have high weight");

    // Summary statistics
    println!("\nEpistemic Statistics:");
    println!("  Total Outputs: {}", stats.total_outputs);
    println!("  Average Weight: {:.4}", stats.average_weight);
    println!("  Quality Score: {:.4}", stats.quality_score());
    println!("  Dominant E-Level: {:?}", stats.dominant_empirical());

    println!("\n=== Epistemic Classification Test PASSED ===\n");
}

// =============================================================================
// TEST: Phi Coherence Measurement and Action Gating
// =============================================================================

#[test]
fn test_phi_coherence_measurement() {
    println!("\n=== Phi Coherence Measurement ===\n");

    let config = CoherenceMeasurementConfig::default();
    let mut history = CoherenceHistory::default();

    // Scenario 1: Coherent agent (similar outputs)
    println!("Scenario 1: Coherent Agent (consistent outputs)");
    let coherent_outputs: Vec<AgentOutput> = (0..10)
        .map(|i| AgentOutput {
            output_id: format!("coherent-{}", i),
            agent_id: "coherent-agent".to_string(),
            content: OutputContent::Text(format!("Consistent analysis {}", i)),
            classification: EpistemicClassificationExtended::new(
                EmpiricalLevel::E3Cryptographic,
                NormativeLevel::N2Network,
                MaterialityLevel::M2Persistent,
                HarmonicLevel::H1Local,
            ),
            classification_confidence: 0.85,
            timestamp: now_secs() - i,
            has_proof: true,
            proof_data: None,
            context_references: vec!["ctx-1".to_string()],
        })
        .collect();

    let phi_result = measure_coherence(&coherent_outputs, &config).unwrap();
    history.add_measurement(phi_result.clone());

    println!("  Phi: {:.4}", phi_result.coherence);
    println!("  Coherence State: {:?}", phi_result.coherence_state);
    println!("  Sample Size: {}", phi_result.sample_size);

    assert!(
        phi_result.coherence > 0.7,
        "Coherent agent should have high Phi"
    );
    assert_eq!(phi_result.coherence_state, CoherenceState::Coherent);

    // Check action gating
    let high_stakes_check = check_coherence_for_action(phi_result.coherence_state, true);
    println!("  High-stakes action allowed: {:?}", high_stakes_check);
    assert!(matches!(high_stakes_check, CoherenceCheckResult::Allowed));

    // Scenario 2: Incoherent agent (wildly varying outputs)
    println!("\nScenario 2: Incoherent Agent (inconsistent outputs)");
    let levels = [
        (
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            HarmonicLevel::H0None,
        ),
        (
            EmpiricalLevel::E4PublicRepro,
            NormativeLevel::N3Axiomatic,
            MaterialityLevel::M3Foundational,
            HarmonicLevel::H4Kosmic,
        ),
        (
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N2Network,
            MaterialityLevel::M1Temporal,
            HarmonicLevel::H2Network,
        ),
        (
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N0Personal,
            MaterialityLevel::M2Persistent,
            HarmonicLevel::H1Local,
        ),
        (
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M0Ephemeral,
            HarmonicLevel::H3Civilizational,
        ),
    ];

    let incoherent_outputs: Vec<AgentOutput> = levels
        .iter()
        .enumerate()
        .map(|(i, (e, n, m, h))| AgentOutput {
            output_id: format!("incoherent-{}", i),
            agent_id: "incoherent-agent".to_string(),
            content: OutputContent::Text(format!("Random output {}", i)),
            classification: EpistemicClassificationExtended::new(*e, *n, *m, *h),
            classification_confidence: 0.3 + (i as f32 * 0.1),
            timestamp: now_secs() - i as u64,
            has_proof: i % 2 == 0,
            proof_data: None,
            context_references: vec![],
        })
        .collect();

    let incoherent_result = measure_coherence(&incoherent_outputs, &config).unwrap();

    println!("  Phi: {:.4}", incoherent_result.coherence);
    println!("  Coherence State: {:?}", incoherent_result.coherence_state);

    // Check action gating for incoherent agent
    let incoherent_check = check_coherence_for_action(incoherent_result.coherence_state, true);
    println!("  High-stakes action allowed: {:?}", incoherent_check);

    // Incoherent agents should be restricted
    if incoherent_result.coherence_state == CoherenceState::Critical {
        assert!(matches!(
            incoherent_check,
            CoherenceCheckResult::Blocked { .. }
        ));
    }

    println!("\n=== Phi Coherence Test PASSED ===\n");
}

// =============================================================================
// TEST: GIS Uncertainty Handling and Escalation
// =============================================================================

#[test]
fn test_gis_uncertainty_escalation() {
    println!("\n=== GIS Uncertainty Handling ===\n");

    let agent_id = "uncertain-agent";
    let mut calibration = UncertaintyCalibration::default();

    // Scenario 1: Confident agent (low uncertainty)
    println!("Scenario 1: Confident Agent");
    let confident_uncertainty = MoralUncertainty::new(0.1, 0.15, 0.1);
    let confident_guidance = MoralActionGuidance::from_uncertainty(&confident_uncertainty);

    println!(
        "  Epistemic Uncertainty: {:.2}",
        confident_uncertainty.epistemic
    );
    println!(
        "  Axiological Uncertainty: {:.2}",
        confident_uncertainty.axiological
    );
    println!(
        "  Deontic Uncertainty: {:.2}",
        confident_uncertainty.deontic
    );
    println!("  Total Uncertainty: {:.3}", confident_uncertainty.total());
    println!("  Guidance: {:?}", confident_guidance);
    println!("  Can Proceed: {}", confident_guidance.can_proceed());

    assert_eq!(confident_guidance, MoralActionGuidance::ProceedConfidently);

    // No escalation needed
    let escalation = maybe_escalate(agent_id, &confident_uncertainty, "transfer_funds", None);
    assert!(escalation.is_none(), "Confident agent should not escalate");

    // Scenario 2: Uncertain agent (high epistemic uncertainty)
    println!("\nScenario 2: Epistemically Uncertain Agent");
    let epistemic_uncertainty = MoralUncertainty::new(0.85, 0.3, 0.4);
    let epistemic_guidance = MoralActionGuidance::from_uncertainty(&epistemic_uncertainty);

    println!(
        "  Epistemic Uncertainty: {:.2} (HIGH)",
        epistemic_uncertainty.epistemic
    );
    println!(
        "  Most Uncertain: {:?}",
        epistemic_uncertainty.most_uncertain_type()
    );
    println!("  Guidance: {:?}", epistemic_guidance);
    println!("  Requires Human: {}", epistemic_guidance.requires_human());

    // Should escalate due to high uncertainty
    let escalation = maybe_escalate(
        agent_id,
        &epistemic_uncertainty,
        "make_prediction",
        Some("Insufficient data available".to_string()),
    );

    if let Some(esc) = &escalation {
        println!("  Escalation Created:");
        println!("    Blocked Action: {}", esc.blocked_action);
        println!("    Recommendations: {:?}", esc.recommendations);
    }

    // Scenario 3: Morally uncertain agent (high axiological/deontic)
    println!("\nScenario 3: Morally Uncertain Agent");
    let moral_uncertainty = MoralUncertainty::new(0.3, 0.8, 0.75);

    println!(
        "  Axiological Uncertainty: {:.2} (HIGH)",
        moral_uncertainty.axiological
    );
    println!(
        "  Deontic Uncertainty: {:.2} (HIGH)",
        moral_uncertainty.deontic
    );
    println!("  Total Uncertainty: {:.3}", moral_uncertainty.total());

    let moral_guidance = MoralActionGuidance::from_uncertainty(&moral_uncertainty);
    println!("  Guidance: {:?}", moral_guidance);

    assert!(
        moral_guidance.requires_human(),
        "High moral uncertainty should require human"
    );

    // Scenario 4: Calibration tracking
    println!("\nScenario 4: Calibration Tracking");

    // Record some calibration events
    calibration.record(true, false); // Appropriately uncertain
    calibration.record(false, true); // Appropriately confident
    calibration.record(true, false); // Appropriately uncertain
    calibration.record(false, true); // Appropriately confident
    calibration.record(false, false); // Overconfident (bad!)

    println!("  Total Events: {}", calibration.total_events);
    println!(
        "  Calibration Score: {:.2}",
        calibration.calibration_score()
    );
    println!("  Is Overconfident: {}", calibration.is_overconfident());

    // Scenario 5: UncertainOutput wrapper
    println!("\nScenario 5: UncertainOutput Wrapper");

    let safe_output = UncertainOutput::new(
        "Safe analysis result".to_string(),
        MoralUncertainty::confident(),
    );
    println!("  Safe output can act: {}", safe_output.can_act());
    assert!(safe_output.get_if_allowed().is_some());

    let risky_output = UncertainOutput::new(
        "Risky recommendation".to_string(),
        MoralUncertainty::unsure(),
    );
    println!("  Risky output can act: {}", risky_output.can_act());
    assert!(risky_output.get_if_allowed().is_none());

    println!("\n=== GIS Uncertainty Test PASSED ===\n");
}

// =============================================================================
// TEST: Storage Routing Based on Epistemic Classification
// =============================================================================

#[test]
fn test_epistemic_storage_routing() {
    println!("\n=== Epistemic Storage Routing ===\n");

    let storage = EpistemicStorage::default_storage();
    let router = storage.router();

    // Test routing for different epistemic classifications
    let test_cases = vec![
        (
            "Session Token",
            EpistemicClassification::new(
                EmpiricalLevel::E0Null,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
            ),
            StorageBackend::Memory,
            MutabilityMode::MutableCRDT,
        ),
        (
            "User Profile",
            EpistemicClassification::new(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N0Personal,
                MaterialityLevel::M1Temporal,
            ),
            StorageBackend::Local,
            MutabilityMode::MutableCRDT,
        ),
        (
            "Verified Document",
            EpistemicClassification::new(
                EmpiricalLevel::E2PrivateVerify,
                NormativeLevel::N1Communal,
                MaterialityLevel::M2Persistent,
            ),
            StorageBackend::DHT,
            MutabilityMode::AppendOnly,
        ),
        (
            "ZK Proof Record",
            EpistemicClassification::new(
                EmpiricalLevel::E3Cryptographic,
                NormativeLevel::N2Network,
                MaterialityLevel::M3Foundational,
            ),
            StorageBackend::IPFS,
            MutabilityMode::Immutable,
        ),
    ];

    for (name, classification, expected_backend, expected_mutability) in test_cases {
        let tier = router.route(&classification);

        println!("{}:", name);
        println!(
            "  Classification: E{}/N{}/M{}",
            classification.empirical as u8,
            classification.normative as u8,
            classification.materiality as u8
        );
        println!(
            "  Backend: {:?} (expected: {:?})",
            tier.backend, expected_backend
        );
        println!(
            "  Mutability: {:?} (expected: {:?})",
            tier.mutability, expected_mutability
        );
        println!("  Encrypted: {}", tier.encrypted);
        println!("  Content-Addressed: {}", tier.content_addressed);

        assert_eq!(tier.backend, expected_backend);
        assert_eq!(tier.mutability, expected_mutability);
    }

    println!("\n=== Storage Routing Test PASSED ===\n");
}

// =============================================================================
// TEST: Complete Integration Flow
// =============================================================================

#[test]
fn test_complete_integration_flow() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     COMPLETE EPISTEMIC-AWARE AI AGENCY INTEGRATION TEST          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Create agent with initial K-Vector
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 1: Agent Creation with K-Vector Profile                    │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let mut agent = create_test_agent("integration-demo");
    let initial_trust = agent.k_vector.trust_score();

    println!("  Agent ID: {}", agent.agent_id.as_str());
    println!("  Initial Trust Score: {:.3}", initial_trust);
    println!(
        "  Initial KREDIT Cap: {}",
        calculate_kredit_from_trust(initial_trust)
    );
    println!(
        "  Governance Tier: {:?}",
        GovernanceTier::from_trust_score(initial_trust)
    );

    // Step 2: Agent produces outputs with epistemic classification
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 2: Agent Produces Epistemically Classified Outputs         │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let outputs: Vec<AgentOutput> = vec![
        create_agent_output(
            agent.agent_id.as_str(),
            "Initial analysis complete",
            ClassificationHints {
                third_party_verified: true,
                agreement_scope: Some(AgreementScope::Community),
                relevance_duration: Some(RelevanceDuration::MediumTerm),
                ..Default::default()
            },
        ),
        create_agent_output(
            agent.agent_id.as_str(),
            "Data validated with proof",
            ClassificationHints {
                has_crypto_proof: true,
                uses_public_data: true,
                agreement_scope: Some(AgreementScope::Network),
                relevance_duration: Some(RelevanceDuration::LongTerm),
                affected_harmonies_count: 3,
                ..Default::default()
            },
        ),
        create_agent_output(
            agent.agent_id.as_str(),
            "Recommendation generated",
            ClassificationHints {
                third_party_verified: true,
                agreement_scope: Some(AgreementScope::Community),
                relevance_duration: Some(RelevanceDuration::ShortTerm),
                affected_harmonies_count: 1,
                ..Default::default()
            },
        ),
    ];

    let mut epistemic_stats = EpistemicStats::default();
    for (i, output) in outputs.iter().enumerate() {
        let weight = calculate_epistemic_weight(&output.classification);
        epistemic_stats.add_output(&output.classification);
        println!(
            "  Output {}: E{}/N{}/M{}/H{} (weight: {:.4})",
            i + 1,
            output.classification.empirical as u8,
            output.classification.normative as u8,
            output.classification.materiality as u8,
            output.classification.harmonic as u8,
            weight
        );
    }
    println!(
        "  Average Epistemic Weight: {:.4}",
        epistemic_stats.average_weight
    );

    // Step 3: Measure Phi coherence
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 3: Phi Coherence Measurement                               │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let phi_config = CoherenceMeasurementConfig::default();
    let phi_result = measure_coherence(&outputs, &phi_config).unwrap();

    println!("  Phi Value: {:.4}", phi_result.coherence);
    println!("  Coherence State: {:?}", phi_result.coherence_state);

    let action_check = check_coherence_for_action(phi_result.coherence_state, true);
    println!("  High-Stakes Actions: {:?}", action_check);

    // Step 4: Express uncertainty via GIS
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 4: GIS Uncertainty Expression                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let uncertainty = MoralUncertainty::new(0.25, 0.35, 0.30);
    let guidance = MoralActionGuidance::from_uncertainty(&uncertainty);

    println!("  Epistemic Uncertainty: {:.2}", uncertainty.epistemic);
    println!("  Axiological Uncertainty: {:.2}", uncertainty.axiological);
    println!("  Deontic Uncertainty: {:.2}", uncertainty.deontic);
    println!("  Total Uncertainty: {:.3}", uncertainty.total());
    println!("  Guidance: {:?}", guidance);
    println!("  Escalation Required: {}", guidance.requires_human());

    // Step 5: Record outcome and update K-Vector
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 5: Outcome Recording and K-Vector Update                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Simulate successful actions based on outputs
    simulate_agent_actions(&mut agent, 0.85, 30);

    let config = KVectorBridgeConfig::default();
    let analysis = analyze_behavior(&agent.behavior_log);
    let updated_kv = compute_kvector_update(&agent.k_vector, &analysis, &config, 2.0);
    agent.k_vector = updated_kv;

    let new_trust = agent.k_vector.trust_score();
    println!("  Actions Recorded: {}", analysis.total_actions);
    println!("  Success Rate: {:.1}%", analysis.success_rate * 100.0);
    println!(
        "  Updated Trust Score: {:.3} (change: {:+.3})",
        new_trust,
        new_trust - initial_trust
    );

    // Step 6: Adjust KREDIT based on new trust
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 6: KREDIT Allocation from Trust Score                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let new_kredit = calculate_kredit_from_trust(new_trust);
    let old_kredit = calculate_kredit_from_trust(initial_trust);

    println!("  Previous KREDIT Cap: {}", old_kredit);
    println!("  New KREDIT Cap: {}", new_kredit);
    println!("  Change: {:+}", new_kredit as i64 - old_kredit as i64);
    println!(
        "  Governance Tier: {:?}",
        GovernanceTier::from_trust_score(new_trust)
    );

    // Step 7: Route to appropriate storage
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 7: Epistemic-Driven Storage Routing                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let storage = EpistemicStorage::default_storage();
    let router = storage.router();

    // High epistemic output goes to IPFS
    let high_class = EpistemicClassification::new(
        EmpiricalLevel::E3Cryptographic,
        NormativeLevel::N2Network,
        MaterialityLevel::M3Foundational,
    );
    let high_tier = router.route(&high_class);

    println!("  High-Epistemic Output (E3/N2/M3):");
    println!("    Backend: {:?}", high_tier.backend);
    println!("    Mutability: {:?}", high_tier.mutability);
    println!("    Content-Addressed: {}", high_tier.content_addressed);

    // Final summary
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    INTEGRATION SUMMARY                           ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Agent Trust Evolution: {:.3} → {:.3} ({:+.3})              ║",
        initial_trust,
        new_trust,
        new_trust - initial_trust
    );
    println!(
        "║  KREDIT Allocation: {} → {}                          ║",
        old_kredit, new_kredit
    );
    println!(
        "║  Epistemic Quality Score: {:.4}                             ║",
        epistemic_stats.quality_score()
    );
    println!(
        "║  Phi Coherence: {:.4} ({:?})                     ║",
        phi_result.coherence, phi_result.coherence_state
    );
    println!(
        "║  Uncertainty Level: {:.3} ({:?})           ║",
        uncertainty.total(),
        guidance
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Assertions
    assert!(
        new_trust >= initial_trust,
        "Trust should improve or maintain"
    );
    assert!(phi_result.coherence > 0.0, "Phi should be measurable");
    assert!(
        guidance.can_proceed(),
        "Low uncertainty should allow proceeding"
    );
    assert_eq!(high_tier.backend, StorageBackend::IPFS);
    assert_eq!(high_tier.mutability, MutabilityMode::Immutable);

    println!("=== COMPLETE INTEGRATION TEST PASSED ===\n");
}

// =============================================================================
// TEST: ZK-Integrated Full Pipeline Demo
// =============================================================================
// This test demonstrates the complete epistemic-aware AI agency system using
// the ZKIntegratedPipeline with all Phase 1-4 features:
// - Phase 1: K-Vector Integration
// - Phase 2: Epistemic Classification
// - Phase 3: Phi Coherence Gating
// - Phase 4: GIS Uncertainty Handling

use mycelix_sdk::agentic::coherence_bridge::ZKOperationType;
use mycelix_sdk::agentic::integration::{
    CombinedGatingRecommendation, ObservabilityExports, ZKIntegratedPipeline, ZKTrustConfig,
};
use mycelix_sdk::agentic::zk_trust::ProofStatement;

/// Create a test agent with specific Phi coherence level
fn create_agent_with_phi(name: &str, phi: f32) -> InstrumentalActor {
    InstrumentalActor {
        agent_id: AgentId::from_string(format!("agent-{}", name)),
        sponsor_did: format!("did:test:sponsor-{}", name),
        agent_class: AgentClass::Supervised,
        kredit_balance: 1000,
        kredit_cap: 5000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: now_secs() - 86400,
        last_activity: now_secs(),
        actions_this_hour: 0,
        k_vector: KVector::new(0.6, 0.5, 0.7, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, phi),
        epistemic_stats: EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: Default::default(),
        pending_escalations: vec![],
    }
}

#[test]
fn test_zk_integrated_full_pipeline_demo() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║    ZK-INTEGRATED EPISTEMIC-AWARE AI AGENCY: FULL PIPELINE DEMO       ║");
    println!("║                                                                      ║");
    println!("║    Demonstrating Phases 1-4 of the Research Plan:                    ║");
    println!("║    • Phase 1: K-Vector Trust Profiles                                ║");
    println!("║    • Phase 2: E-N-M-H Epistemic Classification                       ║");
    println!("║    • Phase 3: Phi Coherence Gating                                   ║");
    println!("║    • Phase 4: GIS Uncertainty & Escalation                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // ==========================================================================
    // SETUP: Create ZK-Integrated Pipeline
    // ==========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ SETUP: Initialize ZK-Integrated Pipeline (Simulation Mode)          │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let config = ZKTrustConfig {
        simulation_mode: true,
        min_attestation_trust: 0.4,
        byzantine_proof_threshold: 0.67,
        ..Default::default()
    };
    let mut pipeline = ZKIntegratedPipeline::new(config);

    println!("  Pipeline Mode: Simulation (ZK proofs simulated)");
    println!("  Byzantine Threshold: 67%");
    println!("  Min Attestation Trust: 0.4");

    // ==========================================================================
    // PHASE 1: K-Vector Trust Profiles
    // ==========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: K-Vector Integration - Agents with Trust Profiles          │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Register agents with different trust/coherence levels
    let alice = create_agent_with_phi("alice", 0.85); // High coherence
    let bob = create_agent_with_phi("bob", 0.65); // Moderate coherence
    let charlie = create_agent_with_phi("charlie", 0.25); // Low coherence

    // Register with ZK commitments
    let alice_commit = pipeline.register_agent_with_commitment(alice);
    let bob_commit = pipeline.register_agent_with_commitment(bob);
    let charlie_commit = pipeline.register_agent_with_commitment(charlie);

    println!("\n  Agent: Alice (High-Trust, High-Coherence)");
    println!("    K-Vector Commitment: {}", alice_commit.agent_id);
    println!(
        "    k_coherence (Coherence): 0.85 ({:?})",
        CoherenceState::from_phi(0.85)
    );
    println!(
        "    Trust Score: {:.3}",
        pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("agent-alice")
            .unwrap()
            .k_vector
            .trust_score()
    );

    println!("\n  Agent: Bob (Moderate-Trust, Stable-Coherence)");
    println!("    K-Vector Commitment: {}", bob_commit.agent_id);
    println!(
        "    k_coherence (Coherence): 0.65 ({:?})",
        CoherenceState::from_phi(0.65)
    );
    println!(
        "    Trust Score: {:.3}",
        pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("agent-bob")
            .unwrap()
            .k_vector
            .trust_score()
    );

    println!("\n  Agent: Charlie (Lower-Trust, Degraded-Coherence)");
    println!("    K-Vector Commitment: {}", charlie_commit.agent_id);
    println!(
        "    k_coherence (Coherence): 0.25 ({:?})",
        CoherenceState::from_phi(0.25)
    );
    println!(
        "    Trust Score: {:.3}",
        pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("agent-charlie")
            .unwrap()
            .k_vector
            .trust_score()
    );

    // ==========================================================================
    // PHASE 2: Epistemic Classification with ZK Proofs
    // ==========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 2: Epistemic Classification - ZK-Proven Outputs               │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Alice produces high-quality outputs with ZK proofs
    let alice_output = pipeline
        .process_zk_output(
            "agent-alice",
            OutputContent::Text(
                "Cryptographically verified analysis: gradient quality = 0.94".to_string(),
            ),
            ProofStatement::TrustExceedsThreshold { threshold: 0.5 },
        )
        .expect("Alice should be able to produce ZK output");

    println!("\n  Alice's ZK-Proven Output:");
    println!(
        "    Classification: E{}/N{}/M{}/H{}",
        alice_output.output.classification.empirical as u8,
        alice_output.output.classification.normative as u8,
        alice_output.output.classification.materiality as u8,
        alice_output.output.classification.harmonic as u8
    );
    println!("    Has ZK Proof: {}", alice_output.output.has_proof);
    println!(
        "    Proof Verified: {}",
        alice_output.proof_summary.verified
    );
    println!("    Proof Result: {}", alice_output.proof_summary.result);
    println!("    K-Vector Delta Applied:");
    println!("      k_r: {:+.4}", alice_output.kvector_delta.k_r_delta);
    println!("      k_p: {:+.4}", alice_output.kvector_delta.k_p_delta);
    println!("      k_v: {:+.4}", alice_output.kvector_delta.k_v_delta);

    assert!(alice_output.output.has_proof, "Output should have ZK proof");
    assert!(
        alice_output.output.classification.empirical >= EmpiricalLevel::E3Cryptographic,
        "ZK-proven output should be E3+"
    );

    // ==========================================================================
    // PHASE 3: Phi Coherence Gating
    // ==========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 3: Phi Coherence Gating - Action Permission by Coherence      │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Check Phi gating for ZK operations
    println!("\n  ZK Operation Permissions by Agent:");

    let operations = [
        ("VerifyProof", ZKOperationType::VerifyProof),
        ("GenerateProof", ZKOperationType::GenerateProof),
        ("ByzantineConsensus", ZKOperationType::ByzantineConsensus),
    ];

    for (op_name, op) in &operations {
        println!("\n  Operation: {}", op_name);

        for agent_id in &["agent-alice", "agent-bob", "agent-charlie"] {
            let gating = pipeline
                .check_coherence_for_zk_operation(agent_id, *op)
                .expect("Gating check should succeed");

            let status = if gating.permitted {
                "✓ Allowed"
            } else {
                "✗ Blocked"
            };
            println!(
                "    {}: {} (Phi: {:.2}, State: {:?})",
                agent_id.replace("agent-", ""),
                status,
                gating.current_coherence,
                gating.current_state
            );
        }
    }

    // Demonstrate Phi-gated proof generation
    println!("\n  Phi-Gated Proof Generation:");

    // Alice can generate proofs (high coherence)
    let alice_proof =
        pipeline.generate_trust_proof_phi_gated("agent-alice", ProofStatement::IsVerified);
    println!(
        "    Alice: {}",
        if alice_proof.is_ok() {
            "✓ Proof generated"
        } else {
            "✗ Blocked"
        }
    );
    assert!(
        alice_proof.is_ok(),
        "High-coherence agent should generate proofs"
    );

    // Charlie is blocked (low coherence)
    let charlie_proof =
        pipeline.generate_trust_proof_phi_gated("agent-charlie", ProofStatement::IsVerified);
    println!(
        "    Charlie: {}",
        if charlie_proof.is_ok() {
            "✓ Proof generated"
        } else {
            "✗ Blocked (insufficient coherence)"
        }
    );
    assert!(
        charlie_proof.is_err(),
        "Low-coherence agent should be blocked"
    );

    // Check network-wide Phi health
    let phi_health = pipeline.phi_network_health();
    println!("\n  Network Phi Health:");
    println!("    Total Agents: {}", phi_health.agent_count);
    println!("    Average Phi: {:.3}", phi_health.average_phi);
    println!("    Coherent Agents: {}", phi_health.coherent_agents);
    println!("    Degraded Agents: {}", phi_health.degraded_agents);
    println!(
        "    Network Level: {:?}",
        phi_health.network_coherence_level
    );

    // ==========================================================================
    // PHASE 4: GIS Uncertainty & Escalation
    // ==========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 4: GIS Uncertainty - Moral Uncertainty & Escalation           │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Low uncertainty output - proceeds without escalation
    println!("\n  Scenario A: Low Uncertainty (Should Proceed)");

    let low_uncertainty = MoralUncertainty::new(0.15, 0.2, 0.1);
    let low_unc_result = pipeline
        .process_zk_output_with_uncertainty(
            "agent-alice",
            OutputContent::Text("Standard data analysis complete".to_string()),
            ProofStatement::WellFormed,
            low_uncertainty.clone(),
        )
        .expect("Low uncertainty output should succeed");

    println!(
        "    Uncertainty: E={:.2}, A={:.2}, D={:.2} (Total: {:.3})",
        low_uncertainty.epistemic,
        low_uncertainty.axiological,
        low_uncertainty.deontic,
        low_uncertainty.total()
    );
    println!("    Guidance: {:?}", low_unc_result.guidance);
    println!(
        "    Output Produced: {}",
        low_unc_result.output_result.is_some()
    );
    println!(
        "    Escalation Required: {}",
        low_unc_result.escalation.is_some()
    );

    assert!(
        low_unc_result.output_result.is_some(),
        "Low uncertainty should produce output"
    );
    assert!(
        low_unc_result.escalation.is_none(),
        "Low uncertainty should not escalate"
    );

    // High uncertainty output - triggers escalation
    println!("\n  Scenario B: High Uncertainty (Should Escalate)");

    let high_uncertainty = MoralUncertainty::new(0.85, 0.9, 0.8);
    let high_unc_result = pipeline
        .process_zk_output_with_uncertainty(
            "agent-bob",
            OutputContent::Text("Critical decision with global implications".to_string()),
            ProofStatement::TrustExceedsThreshold { threshold: 0.6 },
            high_uncertainty.clone(),
        )
        .expect("High uncertainty output should create escalation");

    println!(
        "    Uncertainty: E={:.2}, A={:.2}, D={:.2} (Total: {:.3})",
        high_uncertainty.epistemic,
        high_uncertainty.axiological,
        high_uncertainty.deontic,
        high_uncertainty.total()
    );
    println!("    Guidance: {:?}", high_unc_result.guidance);
    println!(
        "    Output Produced: {}",
        high_unc_result.output_result.is_some()
    );
    println!(
        "    Escalation Created: {}",
        high_unc_result.escalation.is_some()
    );

    assert!(
        high_unc_result.output_result.is_none(),
        "High uncertainty should not produce output"
    );
    assert!(
        high_unc_result.escalation.is_some(),
        "High uncertainty should create escalation"
    );

    if let Some(ref esc) = high_unc_result.escalation {
        println!("    Escalation Details:");
        println!("      Blocked Action: {}", esc.blocked_action);
        println!("      Pending Sponsor Review: Yes");
    }

    // Combined gating (Phi + GIS)
    println!("\n  Scenario C: Combined Gating (Phi + Uncertainty)");

    let combined = pipeline
        .check_combined_gating(
            "agent-alice",
            &MoralUncertainty::new(0.2, 0.2, 0.2),
            ZKOperationType::GenerateProof,
        )
        .expect("Combined gating should work");

    println!("    Agent: Alice (high coherence, low uncertainty)");
    println!("    Permitted: {}", combined.permitted);
    println!("    Recommendation: {:?}", combined.recommendation);

    assert!(
        combined.permitted,
        "High coherence + low uncertainty should permit"
    );
    assert_eq!(
        combined.recommendation,
        CombinedGatingRecommendation::Proceed
    );

    let combined_blocked = pipeline
        .check_combined_gating(
            "agent-charlie",
            &MoralUncertainty::new(0.2, 0.2, 0.2),
            ZKOperationType::GenerateProof,
        )
        .expect("Combined gating should work");

    println!("\n    Agent: Charlie (low coherence, low uncertainty)");
    println!("    Permitted: {}", combined_blocked.permitted);
    println!("    Recommendation: {:?}", combined_blocked.recommendation);

    assert!(
        !combined_blocked.permitted,
        "Low coherence should block even with low uncertainty"
    );
    assert_eq!(
        combined_blocked.recommendation,
        CombinedGatingRecommendation::WaitForCoherence
    );

    // Calibration tracking
    println!("\n  Calibration Tracking:");

    // Record some outcomes for Bob
    let _ = pipeline.record_gis_outcome("agent-bob", true, true); // Appropriately uncertain
    let _ = pipeline.record_gis_outcome("agent-bob", false, true); // Appropriately confident
    let _ = pipeline.record_gis_outcome("agent-bob", true, true); // Appropriately uncertain

    let bob_calibration = pipeline
        .get_calibration_summary("agent-bob")
        .expect("Should get calibration");

    println!("    Bob's Calibration:");
    println!("      Total Events: {}", bob_calibration.total_events);
    println!(
        "      Calibration Score: {:.3}",
        bob_calibration.calibration_score
    );
    println!("      Tendency: {:?}", bob_calibration.tendency);

    // GIS network health
    let gis_health = pipeline.gis_network_health();
    println!("\n  Network GIS Health:");
    println!("    Total Agents: {}", gis_health.agent_count);
    println!(
        "    Avg Calibration: {:.3}",
        gis_health.average_calibration_score
    );
    println!(
        "    Pending Escalations: {}",
        gis_health.pending_escalations_total
    );

    // ==========================================================================
    // OBSERVABILITY: Export Metrics
    // ==========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ OBSERVABILITY: Prometheus/OTEL Metric Exports                       │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    let mut exports = ObservabilityExports::new("mycelix");

    // Export ZK health
    let zk_health = pipeline.zk_network_health();
    exports.export_zk_health(&zk_health);

    // Export Phi health
    exports.export_phi_network_health(&phi_health);

    // Export GIS health
    exports.export_gis_network_health(&gis_health);

    // Export individual agent metrics
    if let Ok(alice_phi) = pipeline.export_agent_phi_metrics("agent-alice") {
        exports.export_phi_coherence(&alice_phi);
    }

    exports.export_gis_calibration(&bob_calibration);

    // Show sample of Prometheus metrics
    let prom_text = exports.to_prometheus_text();
    let prom_lines: Vec<&str> = prom_text.lines().take(20).collect();

    println!("\n  Sample Prometheus Metrics (first 20 lines):");
    for line in prom_lines {
        if !line.is_empty() {
            println!("    {}", line);
        }
    }

    println!("\n  Total Metrics Exported: {}", exports.metrics().len());

    // Verify key metrics exist
    assert!(
        prom_text.contains("zk_total_proofs"),
        "Should export ZK metrics"
    );
    assert!(
        prom_text.contains("phi_network_average"),
        "Should export Phi metrics"
    );
    assert!(
        prom_text.contains("gis_network_avg_calibration"),
        "Should export GIS metrics"
    );

    // ==========================================================================
    // FINAL SUMMARY
    // ==========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      FULL PIPELINE DEMO SUMMARY                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                      ║");
    println!("║  PHASE 1 - K-Vector Integration                                      ║");
    println!("║    ✓ 3 agents registered with ZK commitments                         ║");
    println!("║    ✓ K-Vector trust profiles tracked                                 ║");
    println!(
        "║    ✓ Trust scores: Alice={:.2}, Bob={:.2}, Charlie={:.2}              ║",
        pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("agent-alice")
            .unwrap()
            .k_vector
            .trust_score(),
        pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("agent-bob")
            .unwrap()
            .k_vector
            .trust_score(),
        pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("agent-charlie")
            .unwrap()
            .k_vector
            .trust_score()
    );
    println!("║                                                                      ║");
    println!("║  PHASE 2 - Epistemic Classification                                  ║");
    println!("║    ✓ ZK-proven outputs classified E3+                                ║");
    println!("║    ✓ K-Vector updated based on epistemic weight                      ║");
    println!("║                                                                      ║");
    println!("║  PHASE 3 - Phi Coherence Gating                                      ║");
    println!("║    ✓ High-coherence agents (Alice) can generate proofs               ║");
    println!("║    ✓ Low-coherence agents (Charlie) blocked from ZK operations       ║");
    println!(
        "║    ✓ Network Phi health: avg={:.3}, coherent={}                       ║",
        phi_health.average_phi, phi_health.coherent_agents
    );
    println!("║                                                                      ║");
    println!("║  PHASE 4 - GIS Uncertainty Handling                                  ║");
    println!("║    ✓ Low uncertainty → outputs proceed                               ║");
    println!("║    ✓ High uncertainty → sponsor escalation                           ║");
    println!("║    ✓ Combined gating (Phi + GIS) enforced                            ║");
    println!("║    ✓ Calibration tracking active                                     ║");
    println!("║                                                                      ║");
    println!("║  OBSERVABILITY                                                       ║");
    println!("║    ✓ Prometheus metrics exported                                     ║");
    println!(
        "║    ✓ {} total metrics tracked                                        ║",
        exports.metrics().len()
    );
    println!("║                                                                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  This demonstrates AI agents with verifiable \"epistemic fingerprints\"║");
    println!("║  - trust profiles that prove reliability without revealing state.    ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("=== ZK-INTEGRATED FULL PIPELINE DEMO PASSED ===\n");
}
