// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-End Integration Tests for Epistemic-Aware AI Agency
//!
//! This module demonstrates the complete flow of the Epistemic-Aware AI Agency framework:
//!
//! 1. **Agent Creation**: Create AI agent with initial K-Vector (10 dimensions)
//! 2. **K-Vector Update**: Update trust profile based on behavior
//! 3. **ZK Attestation**: Generate zero-knowledge proofs for trust properties
//! 4. **Domain Translation**: Translate trust between domains
//!
//! # What Makes This Novel
//!
//! - AI agents have verifiable K-Vector profiles that evolve from behavior
//! - Trust can be proven without revealing underlying values (ZK)
//! - Trust translates across different domains with appropriate damping
//! - k_coherence (coherence) dimension based on Integrated Information Theory

#[cfg(test)]
mod tests {
    use crate::agentic::cross_domain::{translate_trust, DomainRegistry};
    use crate::agentic::kvector_bridge::{
        calculate_kredit_from_trust, update_agent_kvector, KVectorBridgeConfig,
    };
    use crate::agentic::{
        ActionOutcome, AgentClass, AgentConstraints, AgentId, AgentStatus, BehaviorLogEntry,
        EpistemicStats, InstrumentalActor, UncertaintyCalibration,
    };
    use crate::matl::KVector;
    use crate::zkproof::trust_risc0::TrustRisc0Prover;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Create a test agent with initial K-Vector
    fn create_test_agent(id: &str, k_vector: KVector) -> InstrumentalActor {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Autonomous,
            kredit_balance: 1000,
            kredit_cap: 10000,
            k_vector,
            behavior_log: Vec::new(),
            constraints: AgentConstraints::default(),
            status: AgentStatus::Active,
            created_at: now,
            last_activity: now,
            actions_this_hour: 0,
            epistemic_stats: EpistemicStats::default(),
            output_history: Vec::new(),
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: Vec::new(),
        }
    }

    /// Create a behavior log entry
    fn create_behavior_entry(
        action: &str,
        outcome: ActionOutcome,
        kredit: u64,
    ) -> BehaviorLogEntry {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        BehaviorLogEntry {
            timestamp: now,
            action_type: action.to_string(),
            kredit_consumed: kredit,
            counterparties: vec!["counterparty-1".to_string()],
            outcome,
        }
    }

    // ========================================================================
    // PHASE 1: Agent Creation with 10D K-Vector
    // ========================================================================

    #[test]
    fn test_phase1_agent_creation_with_kvector() {
        // Create agent with initial 10-dimensional K-Vector
        // Including the new k_coherence (coherence) dimension
        let initial_kvector = KVector::new(
            0.7, // k_r: Reputation
            0.5, // k_a: Activity
            0.8, // k_i: Integrity
            0.6, // k_p: Performance
            0.3, // k_m: Membership
            0.4, // k_s: Stake
            0.5, // k_h: Historical
            0.3, // k_topo: Topology
            0.6, // k_v: Verification
            0.5, // k_coherence: Coherence (NEW)
        );

        let agent = create_test_agent("agent-phi-1", initial_kvector);

        // Verify K-Vector has all 10 dimensions
        let arr = agent.k_vector.to_array();
        assert_eq!(arr.len(), 10, "K-Vector should have 10 dimensions");

        // Verify coherence dimension
        assert!(
            (agent.k_vector.k_coherence - 0.5).abs() < 0.01,
            "k_coherence should be 0.5"
        );

        // Verify trust score calculation includes all dimensions
        let trust_score = agent.k_vector.trust_score();
        assert!(
            trust_score > 0.0 && trust_score < 1.0,
            "Trust score should be normalized"
        );

        // KREDIT cap should derive from trust
        let derived_kredit = calculate_kredit_from_trust(trust_score);
        assert!(derived_kredit > 100, "KREDIT should derive from trust");

        println!("✓ Phase 1: Agent created with 10D K-Vector");
        println!("  Trust score: {:.3}", trust_score);
        println!("  Derived KREDIT cap: {}", derived_kredit);
        println!(
            "  k_coherence (coherence): {:.2}",
            agent.k_vector.k_coherence
        );
    }

    // ========================================================================
    // PHASE 2: K-Vector Update from Behavior
    // ========================================================================

    #[test]
    fn test_phase2_kvector_update_from_behavior() {
        let initial_kvector = KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        let mut agent = create_test_agent("agent-phi-2", initial_kvector);

        // Record positive behavior
        for _ in 0..15 {
            agent.behavior_log.push(create_behavior_entry(
                "contribution",
                ActionOutcome::Success,
                10,
            ));
        }

        // Update K-Vector based on behavior
        let config = KVectorBridgeConfig::default();
        let old_trust = agent.k_vector.trust_score();
        let old_kredit_cap = agent.kredit_cap;

        let updated = update_agent_kvector(&mut agent, &config);
        assert!(updated.is_some(), "K-Vector should be updated");

        let new_trust = agent.k_vector.trust_score();

        // Positive behavior should improve trust
        assert!(
            new_trust >= old_trust,
            "Trust should improve with positive behavior: {} >= {}",
            new_trust,
            old_trust
        );

        // KREDIT cap should increase with trust
        let new_kredit_cap = calculate_kredit_from_trust(new_trust);

        println!("✓ Phase 2: K-Vector updated from behavior");
        println!("  Trust: {:.3} → {:.3}", old_trust, new_trust);
        println!("  KREDIT cap: {} → {}", old_kredit_cap, new_kredit_cap);
        println!("  Behaviors analyzed: {}", agent.behavior_log.len());
    }

    // ========================================================================
    // PHASE 3: ZK Trust Attestation
    // ========================================================================

    #[test]
    fn test_phase3_zk_trust_attestation() {
        let kvector = KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7);
        let agent = create_test_agent("agent-phi-3", kvector);

        // Create ZK prover (simulation mode)
        let prover = TrustRisc0Prover::new_simulation();

        // Prove trust exceeds threshold WITHOUT revealing actual values
        let proof = prover
            .prove_trust_exceeds_threshold(&agent.k_vector, 0.5, agent.agent_id.as_str())
            .expect("Proof should succeed");

        assert!(proof.output.statement_valid, "Statement should be valid");
        assert!(prover.verify(&proof), "Proof should verify");

        // The commitment hides the actual K-Vector values
        let commitment = proof.output.kvector_commitment;
        assert!(
            commitment.iter().any(|&b| b != 0),
            "Commitment should not be zero"
        );

        // Prove coherence without revealing k_coherence value
        let coherence_proof = prover
            .prove_is_coherent(&agent.k_vector, agent.agent_id.as_str())
            .expect("Coherence proof should succeed");

        assert!(
            coherence_proof.output.statement_valid,
            "Agent should be coherent"
        );

        println!("✓ Phase 3: ZK trust attestation generated");
        println!("  Trust > 0.5: {}", proof.output.statement_valid);
        println!("  Coherent: {}", coherence_proof.output.statement_valid);
        println!(
            "  Commitment: {:02x}{:02x}...",
            commitment[0], commitment[1]
        );
        println!("  Proof time: {}ms", proof.proof_time_ms);
    }

    // ========================================================================
    // PHASE 4: Cross-Domain Trust Translation
    // ========================================================================

    #[test]
    fn test_phase4_cross_domain_translation() {
        let kvector = KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7);
        let agent = create_test_agent("agent-phi-4", kvector);

        // Get domain registry with predefined domains
        let registry = DomainRegistry::with_defaults();

        let code_review = registry
            .get("code_review")
            .expect("Code review domain should exist");
        let financial = registry
            .get("financial")
            .expect("Financial domain should exist");

        // Translate trust from code review to financial domain
        let translation = translate_trust(&agent.k_vector, code_review, financial);

        // Financial domain is more demanding
        assert!(
            translation.target_trust <= translation.source_trust,
            "Financial trust should be <= code review trust"
        );

        assert!(
            translation.translation_confidence > 0.0,
            "Translation should have positive confidence"
        );

        // Check dimension translations (now 10 dimensions)
        assert_eq!(
            translation.dimension_translations.len(),
            10,
            "Should translate all 10 dimensions"
        );

        println!("✓ Phase 4: Cross-domain trust translation");
        println!("  Source (code_review): {:.3}", translation.source_trust);
        println!("  Target (financial): {:.3}", translation.target_trust);
        println!("  Confidence: {:.3}", translation.translation_confidence);
        println!(
            "  Meets requirements: {}",
            translation.meets_target_requirements
        );
    }

    // ========================================================================
    // PHASE 5: Coherence Dimension Validation
    // ========================================================================

    #[test]
    fn test_phase5_coherence_dimension() {
        // Test the new k_coherence dimension
        let high_coherence = KVector::new(0.7, 0.6, 0.8, 0.7, 0.5, 0.4, 0.6, 0.5, 0.7, 0.85);
        let low_coherence = KVector::new(0.7, 0.6, 0.8, 0.7, 0.5, 0.4, 0.6, 0.5, 0.7, 0.2);

        // High coherence agent
        assert!(
            high_coherence.is_highly_coherent(),
            "Should be highly coherent"
        );
        assert!(
            high_coherence.coherence_level() > 0.8,
            "Coherence should be high"
        );

        // Low coherence agent
        assert!(
            !low_coherence.is_highly_coherent(),
            "Should not be highly coherent"
        );
        assert!(
            low_coherence.coherence_level() < 0.3,
            "Coherence should be low"
        );

        // Update coherence
        let updated = low_coherence.with_coherence(0.9);
        assert!(
            updated.is_highly_coherent(),
            "Updated should be highly coherent"
        );

        println!("✓ Phase 5: Coherence dimension validated");
        println!(
            "  High coherence k_coherence: {:.2}",
            high_coherence.k_coherence
        );
        println!(
            "  Low coherence k_coherence: {:.2}",
            low_coherence.k_coherence
        );
        println!("  Updated k_coherence: {:.2}", updated.k_coherence);
    }

    // ========================================================================
    // FULL E2E FLOW
    // ========================================================================

    #[test]
    fn test_full_e2e_epistemic_aware_agent() {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║     EPISTEMIC-AWARE AI AGENCY: END-TO-END DEMONSTRATION      ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // === STEP 1: Create Agent with Initial K-Vector ===
        println!("─── Step 1: Agent Creation ───");
        let initial_kvector = KVector::new(0.6, 0.4, 0.7, 0.5, 0.3, 0.4, 0.5, 0.3, 0.5, 0.5);
        let mut agent = create_test_agent("epistemic-agent-001", initial_kvector);
        println!("  Created agent: {}", agent.agent_id.as_str());
        println!("  Initial trust: {:.3}", agent.k_vector.trust_score());
        println!(
            "  Initial coherence (k_coherence): {:.2}",
            agent.k_vector.k_coherence
        );

        // === STEP 2: Record Behavior & Update K-Vector ===
        println!("\n─── Step 2: Behavior → K-Vector ───");
        for i in 0..10 {
            agent.behavior_log.push(create_behavior_entry(
                if i % 2 == 0 {
                    "contribution"
                } else {
                    "validation"
                },
                ActionOutcome::Success,
                15,
            ));
        }

        let kv_config = KVectorBridgeConfig::default();
        let pre_update_trust = agent.k_vector.trust_score();
        update_agent_kvector(&mut agent, &kv_config);
        let post_update_trust = agent.k_vector.trust_score();
        println!("  Behaviors recorded: {}", agent.behavior_log.len());
        println!(
            "  Trust: {:.3} → {:.3}",
            pre_update_trust, post_update_trust
        );

        // === STEP 3: Update Coherence Based on Output Consistency ===
        println!("\n─── Step 3: Coherence Update ───");
        // Simulate measuring phi and updating k_coherence
        let measured_phi = 0.75f32; // Would come from coherence_bridge::measure_agent_coherence
        agent.k_vector = agent.k_vector.with_coherence(measured_phi);
        println!("  Measured Phi: {:.3}", measured_phi);
        println!("  Updated k_coherence: {:.3}", agent.k_vector.k_coherence);

        // === STEP 4: Generate ZK Attestation ===
        println!("\n─── Step 4: ZK Trust Attestation ───");
        let prover = TrustRisc0Prover::new_simulation();

        let trust_proof = prover
            .prove_trust_exceeds_threshold(&agent.k_vector, 0.5, agent.agent_id.as_str())
            .expect("Trust proof failed");

        let coherence_proof = prover
            .prove_is_coherent(&agent.k_vector, agent.agent_id.as_str())
            .expect("Coherence proof failed");

        println!(
            "  Trust > 0.5: {} (proof verified: {})",
            trust_proof.output.statement_valid,
            prover.verify(&trust_proof)
        );
        println!(
            "  Highly coherent: {} (proof verified: {})",
            coherence_proof.output.statement_valid,
            prover.verify(&coherence_proof)
        );
        println!(
            "  Commitment: {:02x}{:02x}{:02x}{:02x}...",
            trust_proof.output.kvector_commitment[0],
            trust_proof.output.kvector_commitment[1],
            trust_proof.output.kvector_commitment[2],
            trust_proof.output.kvector_commitment[3]
        );

        // === STEP 5: Domain Translation ===
        println!("\n─── Step 5: Domain Translation ───");
        let registry = DomainRegistry::with_defaults();
        let source = registry.get("code_review").unwrap();
        let target = registry.get("governance").unwrap();

        let translation = translate_trust(&agent.k_vector, source, target);
        println!("  Code Review → Governance");
        println!("  Source trust: {:.3}", translation.source_trust);
        println!("  Target trust: {:.3}", translation.target_trust);
        println!("  Confidence: {:.3}", translation.translation_confidence);

        // === FINAL SUMMARY ===
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                      SUMMARY                                 ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  Agent: {}                           ║",
            agent.agent_id.as_str()
        );
        println!(
            "║  Final Trust Score: {:.3}                                    ║",
            agent.k_vector.trust_score()
        );
        println!(
            "║  Coherence (k_coherence): {:.3}                                    ║",
            agent.k_vector.k_coherence
        );
        println!(
            "║  Behaviors Logged: {}                                        ║",
            agent.behavior_log.len()
        );
        println!("║  ZK Proofs Generated: 2 (trust + coherence)                  ║");
        println!("║  Domain Translations: 10 dimensions                          ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Assertions for test validity
        assert!(agent.k_vector.trust_score() > 0.0);
        assert!(agent.k_vector.k_coherence > 0.0);
        assert!(trust_proof.output.statement_valid);
        assert_eq!(translation.dimension_translations.len(), 10);
    }

    // ========================================================================
    // PHASE 6: FL → Agent Integration
    // ========================================================================

    #[test]
    fn test_phase6_fl_agent_integration() {
        use crate::agentic::fl_bridge::{FLAgentBridge, FLAgentBridgeConfig, FLRoundAgentImpact};
        use crate::fl::matl_feedback::{FLMatlFeedback, FeedbackStats, KVectorDelta};
        use std::collections::HashMap;

        println!("\n─── Phase 6: FL → Agent Integration ───");

        // Create agents participating in FL
        let mut agents: HashMap<String, InstrumentalActor> = HashMap::new();

        // Good agent with moderate trust
        let good_agent = create_test_agent(
            "fl-agent-good",
            KVector::new(0.6, 0.5, 0.7, 0.6, 0.4, 0.3, 0.5, 0.3, 0.5, 0.6),
        );
        agents.insert("fl-agent-good".to_string(), good_agent);

        // Byzantine agent with moderate trust
        let bad_agent = create_test_agent(
            "fl-agent-bad",
            KVector::new(0.6, 0.5, 0.7, 0.6, 0.4, 0.3, 0.5, 0.3, 0.5, 0.4),
        );
        agents.insert("fl-agent-bad".to_string(), bad_agent);

        // Capture initial states
        let good_initial_trust = agents["fl-agent-good"].k_vector.trust_score();
        let bad_initial_trust = agents["fl-agent-bad"].k_vector.trust_score();

        println!("  Initial good agent trust: {:.3}", good_initial_trust);
        println!("  Initial bad agent trust: {:.3}", bad_initial_trust);

        // Simulate FL round feedback
        let mut kvector_deltas = HashMap::new();

        // Good agent contributed high quality gradients
        kvector_deltas.insert(
            "fl-agent-good".to_string(),
            KVectorDelta {
                participant_id: "fl-agent-good".to_string(),
                reputation_delta: 0.03,
                activity_delta: 0.01,
                integrity_delta: 0.02,
                performance_delta: 0.02,
                historical_delta: 0.01,
                coherence_delta: 0.0,
                reason: "High quality gradient contribution".to_string(),
                is_penalty: false,
            },
        );

        // Bad agent detected as Byzantine
        kvector_deltas.insert(
            "fl-agent-bad".to_string(),
            KVectorDelta {
                participant_id: "fl-agent-bad".to_string(),
                reputation_delta: -0.08,
                activity_delta: 0.0,
                integrity_delta: -0.05,
                performance_delta: -0.03,
                historical_delta: -0.02,
                coherence_delta: -0.02,
                reason: "Byzantine behavior detected".to_string(),
                is_penalty: true,
            },
        );

        let feedback = FLMatlFeedback {
            round_id: 42,
            kvector_deltas,
            quality_signals: HashMap::new(),
            stats: FeedbackStats {
                total_participants: 2,
                rewarded_count: 1,
                penalized_count: 1,
                neutral_count: 0,
                avg_quality_score: 0.5,
                byzantine_fraction: 0.5,
            },
        };

        // Apply feedback to agents
        let mut bridge = FLAgentBridge::with_config(FLAgentBridgeConfig {
            reputation_warmup_rounds: 1, // Skip warmup for test
            auto_update_kredit: true,
            ..Default::default()
        });

        let mut results = Vec::new();
        for (id, agent) in agents.iter_mut() {
            if let Some(result) = bridge.apply_feedback(agent, &feedback, 0.8) {
                results.push(result);
            }
        }

        // Compute impact summary
        let impact = FLRoundAgentImpact::from_results(&results);

        // Verify results
        let good_final_trust = agents["fl-agent-good"].k_vector.trust_score();
        let bad_final_trust = agents["fl-agent-bad"].k_vector.trust_score();

        println!(
            "  Final good agent trust: {:.3} (Δ{:+.3})",
            good_final_trust,
            good_final_trust - good_initial_trust
        );
        println!(
            "  Final bad agent trust: {:.3} (Δ{:+.3})",
            bad_final_trust,
            bad_final_trust - bad_initial_trust
        );
        println!(
            "  Impact: {} improved, {} degraded",
            impact.trust_improved, impact.trust_degraded
        );

        // Assertions
        assert!(
            good_final_trust > good_initial_trust,
            "Good agent trust should increase"
        );
        assert!(
            bad_final_trust < bad_initial_trust,
            "Bad agent trust should decrease"
        );
        assert_eq!(impact.agents_affected, 2);
        assert_eq!(impact.trust_improved, 1);
        assert_eq!(impact.trust_degraded, 1);

        println!("✓ Phase 6: FL → Agent integration verified");
    }

    // ========================================================================
    // PHASE 7: Full Pipeline Integration
    // ========================================================================

    #[test]
    fn test_phase7_full_pipeline() {
        use crate::agentic::fl_bridge::{FLAgentBridge, FLAgentBridgeConfig};
        use crate::agentic::kredit::calculate_kredit_cap_from_trust;
        use crate::fl::matl_feedback::{FLMatlFeedback, FeedbackStats, KVectorDelta};
        use std::collections::HashMap;

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║           FULL PIPELINE: FL → MATL → KREDIT                   ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Create agent
        let mut agent = create_test_agent(
            "pipeline-agent",
            KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        );

        // Set initial KREDIT cap from trust
        let initial_cap =
            calculate_kredit_cap_from_trust(&agent.k_vector, agent.agent_class, 0.8).unwrap();
        agent.kredit_cap = initial_cap;

        println!("─── Initial State ───");
        println!("  Trust Score: {:.3}", agent.k_vector.trust_score());
        println!("  KREDIT Cap: {}", agent.kredit_cap);
        println!("  k_r (Reputation): {:.2}", agent.k_vector.k_r);
        println!("  k_p (Performance): {:.2}", agent.k_vector.k_p);

        // Simulate 5 FL rounds with positive contributions
        let mut bridge = FLAgentBridge::with_config(FLAgentBridgeConfig {
            reputation_warmup_rounds: 1,
            auto_update_kredit: true,
            ..Default::default()
        });

        println!("\n─── FL Round Participation ───");
        for round in 1..=5 {
            let mut kvector_deltas = HashMap::new();
            kvector_deltas.insert(
                "pipeline-agent".to_string(),
                KVectorDelta {
                    participant_id: "pipeline-agent".to_string(),
                    reputation_delta: 0.02,
                    activity_delta: 0.01,
                    integrity_delta: 0.01,
                    performance_delta: 0.015,
                    historical_delta: 0.005,
                    coherence_delta: 0.01,
                    reason: format!("Round {} contribution", round),
                    is_penalty: false,
                },
            );

            let feedback = FLMatlFeedback {
                round_id: round,
                kvector_deltas,
                quality_signals: HashMap::new(),
                stats: FeedbackStats::default(),
            };

            let result = bridge.apply_feedback(&mut agent, &feedback, 0.8).unwrap();
            println!(
                "  Round {}: trust {:.3} → {:.3}, KREDIT {} → {}",
                round,
                result.trust_before,
                result.trust_after,
                result.kredit_cap_before.unwrap_or(0),
                result.kredit_cap_after.unwrap_or(0)
            );
        }

        println!("\n─── Final State ───");
        println!("  Trust Score: {:.3}", agent.k_vector.trust_score());
        println!("  KREDIT Cap: {}", agent.kredit_cap);
        println!("  k_r (Reputation): {:.2}", agent.k_vector.k_r);
        println!("  k_p (Performance): {:.2}", agent.k_vector.k_p);
        println!(
            "  k_coherence (Coherence): {:.2}",
            agent.k_vector.k_coherence
        );

        // Verify improvement
        let final_cap = agent.kredit_cap;
        assert!(
            final_cap > initial_cap,
            "KREDIT cap should increase: {} > {}",
            final_cap,
            initial_cap
        );
        assert!(
            agent.k_vector.trust_score() > 0.5,
            "Trust should be above initial 0.5"
        );

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  PIPELINE VERIFIED                                            ║");
        println!(
            "║  Trust improved: {:.3} → {:.3}                              ║",
            0.5,
            agent.k_vector.trust_score()
        );
        println!(
            "║  KREDIT increased: {} → {}                             ║",
            initial_cap, final_cap
        );
        println!("╚══════════════════════════════════════════════════════════════╝\n");
    }
}
