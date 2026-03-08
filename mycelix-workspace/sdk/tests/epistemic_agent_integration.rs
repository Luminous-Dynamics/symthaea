//! # Epistemic-Aware AI Agency Integration Test
//!
//! End-to-end demonstration of the complete Epistemic-Aware AI Agency flow:
//!
//! 1. AI agent created with initial K-Vector
//! 2. Agent produces output with E-N-M-H classification
//! 3. Phi measured for coherence check
//! 4. Uncertainty expressed via GIS
//! 5. Outcome recorded → K-Vector updated
//! 6. KREDIT cap adjusted based on new trust score
//! 7. Full epistemic fingerprint evolution tracked

use mycelix_sdk::agentic::{
    analyze_behavior,
    calculate_epistemic_weight,
    calculate_kredit_from_trust,

    check_coherence_for_action,
    classify_output,
    coherence_to_kvector_dimension,

    compute_kvector_update,
    compute_trust_score,
    maybe_escalate,
    measure_coherence,
    should_proceed,
    update_agent_kvector,
    ActionOutcome,
    AgentClass,
    AgentConstraints,
    // Phase 1: K-Vector Integration
    AgentId,
    // Phase 2: Epistemic Classification
    AgentOutput,
    AgentOutputBuilder,
    AgentStatus,
    AgreementScope,
    BehaviorLogEntry,
    ClassificationHints,
    CoherenceHistory,
    CoherenceMeasurementConfig,
    // Phase 3: Phi Measurement
    CoherenceState,
    EpistemicStats,
    InstrumentalActor,
    KVectorBridgeConfig,
    MoralActionGuidance,
    // Phase 4: GIS Integration
    MoralUncertainty,
    OutputContent,
    RelevanceDuration,

    UncertainOutput,
    UncertaintyCalibration,
};

use mycelix_sdk::agentic::kredit::{calculate_kredit_cap_from_trust, recalculate_agent_kredit_cap};

use mycelix_sdk::epistemic::{
    EmpiricalLevel, EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel,
    NormativeLevel,
};
use mycelix_sdk::matl::KVector;

/// Helper to create an agent for testing
fn create_test_agent(agent_id: &str) -> InstrumentalActor {
    InstrumentalActor {
        agent_id: AgentId::from_string(agent_id.to_string()),
        sponsor_did: "did:test:sponsor".to_string(),
        agent_class: AgentClass::Supervised,
        kredit_balance: 10000,
        kredit_cap: 10000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 86400, // Created 1 day ago
        last_activity: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        actions_this_hour: 0,
        k_vector: KVector::new_participant(),
        epistemic_stats: EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: Default::default(),
        pending_escalations: vec![],
    }
}

/// Helper to create agent outputs
fn create_agent_output(
    agent_id: &str,
    e: EmpiricalLevel,
    n: NormativeLevel,
    m: MaterialityLevel,
    h: HarmonicLevel,
    timestamp: u64,
) -> AgentOutput {
    AgentOutput {
        output_id: format!("out-{}-{}", agent_id, timestamp),
        agent_id: agent_id.to_string(),
        content: OutputContent::Text(format!("Output at {}", timestamp)),
        classification: EpistemicClassificationExtended::new(e, n, m, h),
        classification_confidence: 0.85,
        timestamp,
        has_proof: matches!(
            e,
            EmpiricalLevel::E3Cryptographic | EmpiricalLevel::E4PublicRepro
        ),
        proof_data: None,
        context_references: vec![],
    }
}

#[test]
fn test_full_agent_lifecycle() {
    // =========================================================================
    // STEP 1: Create agent with initial K-Vector
    // =========================================================================
    let mut agent = create_test_agent("agent-demo-001");

    // Verify initial K-Vector state (new participant)
    assert!((agent.k_vector.k_r - 0.5).abs() < 0.01); // Neutral reputation
    assert!((agent.k_vector.k_i - 1.0).abs() < 0.01); // Full integrity
    assert!((agent.k_vector.k_m - 0.0).abs() < 0.01); // Just joined

    let initial_trust = agent.k_vector.trust_score();
    println!("Initial trust score: {:.4}", initial_trust);

    // =========================================================================
    // STEP 2: Agent produces outputs with E-N-M-H classification
    // =========================================================================
    let base_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Create a sequence of high-quality outputs
    let outputs: Vec<AgentOutput> = (0..10)
        .map(|i| {
            create_agent_output(
                agent.agent_id.as_str(),
                EmpiricalLevel::E3Cryptographic, // High verifiability
                NormativeLevel::N2Network,       // Network-wide agreement
                MaterialityLevel::M2Persistent,  // Long-lasting relevance
                HarmonicLevel::H1Local,          // Local resonance
                base_time + i * 60,
            )
        })
        .collect();

    // Calculate epistemic stats
    let mut epistemic_stats = EpistemicStats::default();
    for output in &outputs {
        epistemic_stats.add_output(&output.classification);
    }

    println!(
        "Epistemic quality score: {:.4}",
        epistemic_stats.quality_score()
    );
    assert!(epistemic_stats.quality_score() > 0.5); // Good quality outputs

    // =========================================================================
    // STEP 3: Measure Phi (coherence) for the output sequence
    // =========================================================================
    let phi_config = CoherenceMeasurementConfig::default();
    let phi_result = measure_coherence(&outputs, &phi_config).unwrap();

    println!("Phi measurement: {:.4}", phi_result.coherence);
    println!("Coherence state: {:?}", phi_result.coherence_state);

    // Consistent outputs should have high coherence
    assert!(phi_result.coherence > 0.8);
    assert!(matches!(
        phi_result.coherence_state,
        CoherenceState::Coherent
    ));

    // Check if high-stakes actions are allowed
    let coherence_check = check_coherence_for_action(phi_result.coherence_state, true);
    assert!(coherence_check.can_proceed());

    // =========================================================================
    // STEP 4: Express uncertainty via GIS
    // =========================================================================
    let uncertainty = MoralUncertainty::new(0.2, 0.3, 0.25);
    let guidance = MoralActionGuidance::from_uncertainty(&uncertainty);

    println!(
        "Uncertainty: epistemic={:.2}, axiological={:.2}, deontic={:.2}",
        uncertainty.epistemic, uncertainty.axiological, uncertainty.deontic
    );
    println!("Guidance: {:?}", guidance);

    // Low uncertainty should allow proceeding
    assert!(should_proceed(&uncertainty));
    assert!(maybe_escalate(agent.agent_id.as_str(), &uncertainty, "transfer", None).is_none());

    // =========================================================================
    // STEP 5: Record successful outcomes → Update K-Vector
    // =========================================================================
    // Add successful behavior entries
    for i in 0..20 {
        agent.behavior_log.push(BehaviorLogEntry {
            timestamp: base_time + i * 60,
            action_type: "api_call".to_string(),
            kredit_consumed: 10,
            counterparties: vec![format!("peer_{}", i % 5)],
            outcome: ActionOutcome::Success,
        });
    }

    // Update K-Vector from behavior
    let config = KVectorBridgeConfig::default();
    let updated_kvector = update_agent_kvector(&mut agent, &config);

    assert!(updated_kvector.is_some());
    let new_kvector = updated_kvector.unwrap();

    println!("\nK-Vector evolution:");
    println!("  k_r (reputation):  {:.4} → {:.4}", 0.5, new_kvector.k_r);
    println!("  k_a (activity):    {:.4} → {:.4}", 0.0, new_kvector.k_a);
    println!("  k_i (integrity):   {:.4} → {:.4}", 1.0, new_kvector.k_i);
    println!("  k_p (performance): {:.4} → {:.4}", 0.5, new_kvector.k_p);

    // With 100% success rate, reputation should improve
    assert!(new_kvector.k_r > 0.5);
    // Activity should be non-zero now
    assert!(new_kvector.k_a > 0.0);

    let new_trust = new_kvector.trust_score();
    println!("\nTrust score: {:.4} → {:.4}", initial_trust, new_trust);

    // =========================================================================
    // STEP 6: KREDIT cap adjusts based on trust score
    // =========================================================================
    let sponsor_civ = 0.75;

    // Calculate new KREDIT cap
    let new_cap =
        calculate_kredit_cap_from_trust(&new_kvector, AgentClass::Supervised, sponsor_civ)
            .expect("Should calculate KREDIT");

    println!("New KREDIT cap: {}", new_cap);
    assert!(new_cap > 5000); // Good trust = good KREDIT

    // Update agent's KREDIT cap
    let _ = recalculate_agent_kredit_cap(&mut agent, sponsor_civ);
    assert!(agent.kredit_cap > 0);

    // =========================================================================
    // STEP 7: Track epistemic fingerprint evolution
    // =========================================================================
    let mut coherence_history = CoherenceHistory::default();
    coherence_history.add_measurement(phi_result.clone());

    // Simulate more measurements over time
    for _ in 0..5 {
        let similar_result = mycelix_sdk::agentic::AgentCoherenceResult {
            coherence: phi_result.coherence * 0.98, // Slightly varying
            coherence_state: CoherenceState::Coherent,
            sample_size: 10,
            output_contributions: vec![],
            measured_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        coherence_history.add_measurement(similar_result);
    }

    println!("\nCoherence history:");
    println!("  Rolling Phi: {:.4}", coherence_history.rolling_coherence);
    println!("  Current state: {:?}", coherence_history.current_state());
    println!(
        "  Trend: {} (1=improving, 0=stable, -1=declining)",
        coherence_history.trend
    );

    // Should still be coherent after stable measurements
    assert!(matches!(
        coherence_history.current_state(),
        CoherenceState::Coherent | CoherenceState::Stable
    ));

    // =========================================================================
    // VERIFICATION: Complete epistemic fingerprint
    // =========================================================================
    println!("\n========== EPISTEMIC FINGERPRINT ==========");
    println!("Agent ID: {}", agent.agent_id.as_str());
    println!("Trust Score: {:.4}", agent.k_vector.trust_score());
    println!(
        "Coherence (Phi): {:.4}",
        coherence_history.rolling_coherence
    );
    println!("Epistemic Quality: {:.4}", epistemic_stats.quality_score());
    println!("KREDIT Cap: {}", agent.kredit_cap);
    println!("Status: {:?}", agent.status);
    println!("============================================");
}

#[test]
fn test_agent_degradation_scenario() {
    // Test what happens when an agent behaves badly
    let mut agent = create_test_agent("agent-bad-001");

    let base_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Agent produces low-quality outputs (all E0/N0/M0/H0)
    let bad_outputs: Vec<AgentOutput> = (0..10)
        .map(|i| {
            create_agent_output(
                agent.agent_id.as_str(),
                EmpiricalLevel::E0Null,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
                HarmonicLevel::H0None,
                base_time + i * 60,
            )
        })
        .collect();

    // Low epistemic quality
    let mut stats = EpistemicStats::default();
    for out in &bad_outputs {
        stats.add_output(&out.classification);
    }
    assert!(stats.quality_score() < 0.3);

    // Agent also fails many actions
    for i in 0..20 {
        agent.behavior_log.push(BehaviorLogEntry {
            timestamp: base_time + i * 60,
            action_type: "api_call".to_string(),
            kredit_consumed: 10,
            counterparties: vec![],
            outcome: if i % 3 == 0 {
                ActionOutcome::ConstraintViolation
            } else {
                ActionOutcome::Error
            },
        });
    }

    // Update K-Vector - should degrade
    let config = KVectorBridgeConfig::default();
    let updated = update_agent_kvector(&mut agent, &config).unwrap();

    // Poor behavior should hurt reputation and integrity
    assert!(updated.k_r < 0.5); // Degraded reputation
    assert!(updated.k_i < 1.0); // Integrity hit from violations

    let trust = updated.trust_score();
    println!("Bad agent trust score: {:.4}", trust);

    // KREDIT should be low
    let kredit = calculate_kredit_from_trust(trust);
    println!("Bad agent KREDIT: {}", kredit);
    assert!(kredit < 15000);
}

#[test]
fn test_high_uncertainty_escalation() {
    // Test that high uncertainty triggers escalation
    let agent = create_test_agent("agent-uncertain-001");

    // High uncertainty across all dimensions (total will be > 0.8)
    let uncertainty = MoralUncertainty::new(0.85, 0.85, 0.85);

    let _guidance = MoralActionGuidance::from_uncertainty(&uncertainty);

    // With total uncertainty > 0.8, should require human
    assert!(!should_proceed(&uncertainty)); // Should NOT proceed

    let escalation = maybe_escalate(
        agent.agent_id.as_str(),
        &uncertainty,
        "high_value_transfer",
        Some("Transferring 10000 SAP to unknown recipient".to_string()),
    );

    assert!(escalation.is_some());
    let esc = escalation.unwrap();

    println!("Escalation triggered:");
    println!("  Guidance: {:?}", esc.guidance);
    println!("  Blocked action: {}", esc.blocked_action);
    println!("  Recommendations:");
    for rec in &esc.recommendations {
        println!("    - {}", rec);
    }

    // Should have recommendations from all three types
    assert!(!esc.recommendations.is_empty());
    assert!(esc
        .recommendations
        .iter()
        .any(|r| r.contains("factual") || r.contains("values") || r.contains("action")));
}

#[test]
fn test_diverse_outputs_lower_coherence() {
    let agent_id = "agent-diverse-001";

    let base_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Create wildly different outputs (should have lower Phi)
    let diverse_outputs = vec![
        create_agent_output(
            agent_id,
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            HarmonicLevel::H0None,
            base_time,
        ),
        create_agent_output(
            agent_id,
            EmpiricalLevel::E4PublicRepro,
            NormativeLevel::N3Axiomatic,
            MaterialityLevel::M3Foundational,
            HarmonicLevel::H4Kosmic,
            base_time + 60,
        ),
        create_agent_output(
            agent_id,
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M1Temporal,
            HarmonicLevel::H2Network,
            base_time + 120,
        ),
    ];

    let phi_result =
        measure_coherence(&diverse_outputs, &CoherenceMeasurementConfig::default()).unwrap();

    println!("Diverse outputs Phi: {:.4}", phi_result.coherence);
    println!("Coherence state: {:?}", phi_result.coherence_state);

    // Should be less coherent than consistent outputs
    assert!(phi_result.coherence < 0.8);
}

#[test]
fn test_uncertainty_calibration_tracking() {
    let mut calibration = UncertaintyCalibration::default();

    // Agent is appropriately calibrated
    calibration.record(true, false); // Uncertain when should be (bad outcome avoided)
    calibration.record(false, true); // Confident when should be (good outcome)
    calibration.record(false, true); // Confident when should be (good outcome)
    calibration.record(true, false); // Uncertain when should be (bad outcome avoided)

    println!(
        "Calibration score (good agent): {:.4}",
        calibration.calibration_score()
    );
    assert!(calibration.calibration_score() > 0.8); // Well calibrated

    // Now simulate an overconfident agent
    let mut bad_calibration = UncertaintyCalibration::default();
    for _ in 0..10 {
        bad_calibration.record(false, false); // Confident but wrong!
    }

    println!(
        "Calibration score (overconfident agent): {:.4}",
        bad_calibration.calibration_score()
    );
    assert!(bad_calibration.calibration_score() < 0.2);
    assert!(bad_calibration.is_overconfident());
}

#[test]
fn test_byzantine_gaming_prevention() {
    // Test that an agent can't game the K-Vector by simply claiming success
    let mut agent = create_test_agent("agent-gamer-001");

    let base_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Agent claims 100% success but with many constraint violations
    // This pattern should be detected through low integrity
    for i in 0..30 {
        let outcome = if i % 2 == 0 {
            ActionOutcome::Success
        } else {
            ActionOutcome::ConstraintViolation
        };

        agent.behavior_log.push(BehaviorLogEntry {
            timestamp: base_time + i * 60,
            action_type: "suspicious_action".to_string(),
            kredit_consumed: 100,                          // High KREDIT usage
            counterparties: vec!["same_peer".to_string()], // Always same counterparty
            outcome,
        });
    }

    let config = KVectorBridgeConfig::default();
    let updated = update_agent_kvector(&mut agent, &config).unwrap();

    // Despite some successes, violations should hurt integrity
    assert!(updated.k_i < 0.5);

    // Low topology diversity (only one counterparty)
    assert!(updated.k_topo < 0.5);

    // Overall trust should be limited
    let trust = updated.trust_score();
    println!("Gaming agent trust: {:.4}", trust);
    assert!(trust < 0.6);
}
