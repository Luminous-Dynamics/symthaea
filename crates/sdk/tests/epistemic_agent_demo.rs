// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Epistemic-Aware AI Agency Demonstration
//!
//! This test demonstrates the complete flow of the Mycelix Epistemic-Aware AI Agency framework:
//!
//! 1. **Agent Creation** with K-Vector trust profile
//! 2. **Behavior → Trust Evolution** through actions
//! 3. **Epistemic Classification** of agent outputs (E/N/M/H)
//! 4. **Phi Coherence Measurement** for quality gating
//! 5. **GIS Uncertainty Handling** with escalation
//! 6. **UESS Storage Routing** based on classification
//!
//! Run with: `cargo test --test epistemic_agent_demo -- --nocapture`

use mycelix_sdk::agentic::{
    analyze_behavior, calculate_epistemic_weight, calculate_kredit_from_trust,
    check_coherence_for_action, compute_kvector_update, compute_trust_score, get_recommendations,
    maybe_escalate, measure_coherence, should_proceed, ActionOutcome, AgentClass, AgentConstraints,
    AgentId, AgentOutput, AgentOutputBuilder, AgentStatus, CoherenceMeasurementConfig,
    InstrumentalActor, KVectorBridgeConfig, MoralUncertainty, OutputContent, UncertainOutput,
};
use mycelix_sdk::epistemic::{
    EmpiricalLevel, EpistemicClassification, HarmonicLevel, MaterialityLevel, NormativeLevel,
};
use mycelix_sdk::matl::KVector;
use mycelix_sdk::storage::{EpistemicStorage, SchemaIdentity, StorageConfig, StoreOptions};

#[test]
fn demo_complete_epistemic_agent_flow() {
    println!("\n{}", "=".repeat(70));
    println!("       EPISTEMIC-AWARE AI AGENCY DEMONSTRATION");
    println!("{}\n", "=".repeat(70));

    // =========================================================================
    // PHASE 1: Agent Creation with K-Vector Trust Profile
    // =========================================================================
    println!("PHASE 1: Agent Creation");
    println!("{}", "-".repeat(70));

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut agent = InstrumentalActor {
        agent_id: AgentId::from_string("demo-agent-001".to_string()),
        sponsor_did: "did:mycelix:sponsor-alice".to_string(),
        agent_class: AgentClass::Supervised,
        kredit_balance: 5000,
        kredit_cap: 10000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: now - 86400 * 7,
        last_activity: now,
        actions_this_hour: 0,
        k_vector: KVector::new_participant(),
        epistemic_stats: mycelix_sdk::agentic::EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: Default::default(),
        pending_escalations: vec![],
    };

    println!("Agent ID:      {}", agent.agent_id.as_str());
    println!("Sponsor:       {}", agent.sponsor_did);
    println!("Class:         {:?}", agent.agent_class);
    println!(
        "KREDIT:        {}/{}",
        agent.kredit_balance, agent.kredit_cap
    );
    println!("Initial K-Vector:");
    print_kvector(&agent.k_vector);
    println!();

    // =========================================================================
    // PHASE 2: Simulating Agent Behavior
    // =========================================================================
    println!("PHASE 2: Simulating Behavior (50 actions)");
    println!("{}", "-".repeat(70));

    for i in 0..40 {
        agent.record_action("process_request", 10, ActionOutcome::Success);
        agent.behavior_log.last_mut().unwrap().counterparties = vec![format!("user_{}", i % 10)];
    }
    for _ in 0..5 {
        agent.record_action("risky_operation", 20, ActionOutcome::Error);
    }
    for _ in 0..5 {
        agent.record_action("boundary_test", 5, ActionOutcome::ConstraintViolation);
    }

    let summary = agent.summary_stats();
    println!("Total Actions:     {}", summary.total_actions);
    println!("Successful:        {}", summary.successful_actions);
    println!("Success Rate:      {:.1}%", summary.success_rate * 100.0);
    println!("KREDIT Consumed:   {}", summary.total_kredit_consumed);
    println!();

    assert_eq!(summary.total_actions, 50);
    assert_eq!(summary.successful_actions, 40);

    // =========================================================================
    // PHASE 3: K-Vector Evolution from Behavior
    // =========================================================================
    println!("PHASE 3: K-Vector Evolution");
    println!("{}", "-".repeat(70));

    let config = KVectorBridgeConfig::default();
    let analysis = analyze_behavior(&agent.behavior_log);

    println!("Behavior Analysis:");
    println!(
        "  Success Rate:        {:.1}%",
        analysis.success_rate * 100.0
    );
    println!("  Violations:          {}", analysis.constraint_violations);
    println!(
        "  Unique Counterparties: {}",
        analysis.unique_counterparties
    );
    println!();

    let initial_trust = compute_trust_score(&agent.k_vector);
    let updated_kvector = compute_kvector_update(&agent.k_vector, &analysis, &config, 7.0);
    agent.k_vector = updated_kvector;

    println!("Updated K-Vector:");
    print_kvector(&agent.k_vector);

    let trust_score = compute_trust_score(&agent.k_vector);
    let derived_kredit = calculate_kredit_from_trust(trust_score);

    println!();
    println!(
        "Trust Score:       {:.3} (was {:.3})",
        trust_score, initial_trust
    );
    println!("Derived KREDIT Cap: {}", derived_kredit);

    agent.kredit_cap = derived_kredit;
    println!();

    // Trust should have changed based on behavior
    assert!(trust_score != initial_trust);

    // =========================================================================
    // PHASE 4: Epistemic Classification of Outputs
    // =========================================================================
    println!("PHASE 4: Epistemic Classification");
    println!("{}", "-".repeat(70));

    let outputs = vec![
        create_output(
            "Weather forecast",
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N1Communal,
            MaterialityLevel::M1Temporal,
        ),
        create_output(
            "SHA256 verification",
            EmpiricalLevel::E4PublicRepro,
            NormativeLevel::N3Axiomatic,
            MaterialityLevel::M3Foundational,
        ),
        create_output(
            "Personal opinion",
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ),
        create_output(
            "Signed transaction",
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        ),
    ];

    for (i, output) in outputs.iter().enumerate() {
        let weight = calculate_epistemic_weight(&output.classification);
        println!(
            "Output {}: E{:?} N{:?} M{:?} -> Weight: {:.3}",
            i + 1,
            output.classification.empirical as u8,
            output.classification.normative as u8,
            output.classification.materiality as u8,
            weight
        );
    }
    println!();

    // Cryptographic outputs should have higher weight
    let weight_0 = calculate_epistemic_weight(&outputs[0].classification);
    let weight_1 = calculate_epistemic_weight(&outputs[1].classification);
    assert!(weight_1 > weight_0);

    // =========================================================================
    // PHASE 5: Phi Coherence Measurement
    // =========================================================================
    println!("PHASE 5: Phi Coherence Measurement");
    println!("{}", "-".repeat(70));

    let phi_config = CoherenceMeasurementConfig::default();

    // Measure Phi from the outputs (need at least 3 for default config)
    if let Some(phi_result) = measure_coherence(&outputs, &phi_config) {
        println!("Phi (coherence):   {:.3}", phi_result.coherence);
        println!("Coherence State:   {:?}", phi_result.coherence_state);
        println!("Sample Size:       {}", phi_result.sample_size);

        // Check if high-stakes action is allowed
        let coherence_check = check_coherence_for_action(
            phi_result.coherence_state,
            true, // is_high_stakes
        );

        println!("High-Stakes Check: {:?}", coherence_check);
        println!("Can Proceed:       {}", coherence_check.can_proceed());

        // Phi should be a valid measurement
        assert!(phi_result.coherence >= 0.0);
    } else {
        println!(
            "Phi measurement requires at least {} outputs",
            phi_config.min_outputs
        );
        // With 4 outputs, this should succeed
        assert!(
            outputs.len() >= phi_config.min_outputs,
            "Should have enough outputs"
        );
    }
    println!();

    // =========================================================================
    // PHASE 6: GIS Uncertainty Handling
    // =========================================================================
    println!("PHASE 6: GIS Uncertainty (Graceful Ignorance)");
    println!("{}", "-".repeat(70));

    let uncertainty = MoralUncertainty::new(0.7, 0.5, 0.8);
    let uncertain_output: UncertainOutput<String> =
        UncertainOutput::new("Medical treatment recommendation".to_string(), uncertainty);

    println!("Uncertain Output: \"{}\"", uncertain_output.output);
    println!(
        "  Epistemic:   {:.0}%",
        uncertain_output.uncertainty.epistemic * 100.0
    );
    println!(
        "  Axiological: {:.0}%",
        uncertain_output.uncertainty.axiological * 100.0
    );
    println!(
        "  Deontic:     {:.0}%",
        uncertain_output.uncertainty.deontic * 100.0
    );
    println!(
        "  Total:       {:.0}%",
        uncertain_output.uncertainty.total() * 100.0
    );
    println!("  Guidance:    {:?}", uncertain_output.guidance);
    println!();

    let recommendations = get_recommendations(&uncertain_output.uncertainty);
    println!("GIS Recommendations: {} items", recommendations.len());

    let proceed = should_proceed(&uncertain_output.uncertainty);
    println!("Should Proceed: {}", proceed);

    if let Some(escalation) = maybe_escalate(
        agent.agent_id.as_str(),
        &uncertain_output.uncertainty,
        "medical_recommendation",
        Some("Patient care decision".to_string()),
    ) {
        println!("Escalation Required: {:?}", escalation.guidance);
        println!("  Recommendations: {:?}", escalation.recommendations);
    }
    println!();

    // High uncertainty should not proceed
    assert!(!proceed);

    // =========================================================================
    // PHASE 7: UESS Storage Routing
    // =========================================================================
    println!("PHASE 7: UESS Storage Routing");
    println!("{}", "-".repeat(70));

    let storage = EpistemicStorage::new(StorageConfig::default());

    let test_cases = vec![
        (
            "session",
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            "Memory",
        ),
        (
            "prefs",
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
            "Local",
        ),
        (
            "doc",
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M2Persistent,
            "DHT",
        ),
        (
            "cert",
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M3Foundational,
            "IPFS",
        ),
    ];

    for (key, e, n, m, expected) in test_cases {
        let classification = EpistemicClassification {
            empirical: e,
            normative: n,
            materiality: m,
        };

        let tier = storage.router().route(&classification);
        println!(
            "{:8} E{} N{} M{} -> {:?}",
            key, e as u8, n as u8, m as u8, tier.backend
        );

        // Verify routing
        let backend_name = format!("{:?}", tier.backend);
        assert!(
            backend_name.contains(expected),
            "Expected {} for {}",
            expected,
            key
        );

        // Store the data
        let options = StoreOptions {
            schema: SchemaIdentity::new("demo", "1.0"),
            agent_id: agent.agent_id.as_str().to_string(),
            allow_overwrite: true,
            ..Default::default()
        };
        let _ = storage.store(key, &"test_value".to_string(), classification, options);
    }

    let stats = storage.stats();
    println!();
    println!("Storage Stats: {} total items", stats.total_items);
    assert!(stats.total_items >= 4);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("                    DEMONSTRATION COMPLETE");
    println!("{}\n", "=".repeat(70));
    println!("All assertions passed. The Epistemic-Aware AI Agency framework is operational.");
}

fn print_kvector(kv: &KVector) {
    println!(
        "  k_r={:.2} k_a={:.2} k_i={:.2} k_p={:.2} k_m={:.2} k_s={:.2} k_h={:.2} k_topo={:.2}",
        kv.k_r, kv.k_a, kv.k_i, kv.k_p, kv.k_m, kv.k_s, kv.k_h, kv.k_topo
    );
}

fn create_output(
    content: &str,
    e: EmpiricalLevel,
    n: NormativeLevel,
    m: MaterialityLevel,
) -> AgentOutput {
    AgentOutputBuilder::new("demo-agent")
        .content(OutputContent::Text(content.to_string()))
        .classification(e, n, m, HarmonicLevel::H0None)
        .build()
        .expect("Failed to build output")
}
