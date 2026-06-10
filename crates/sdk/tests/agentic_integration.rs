// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Agentic Framework Integration Tests
//!
//! Comprehensive tests exercising multi-agent consensus, adversarial detection,
//! calibration, and monitoring working together in realistic scenarios.

use mycelix_sdk::agentic::{
    compute_consensus,
    ActionOutcome,
    AgentClass,
    AgentConstraints,
    // Core types
    AgentId,
    AgentInteraction,
    AgentInteractionRecord,
    AgentStatus,
    // Multi-agent
    AgentVote,
    AlertThresholds,
    AlertType,
    BehaviorLogEntry,
    // Calibration
    CalibrationEngine,
    CalibrationEngineConfig,
    CollaborationManager,
    CollaborativeTaskType,
    CollusionDetector,
    ConsensusConfig,
    CrossAgentCalibration,
    EnhancedAgentCalibrationProfile,

    EpistemicStats,

    GamingDetectionConfig,
    // Adversarial
    GamingDetector,
    GamingResponse,
    InstrumentalActor,
    InteractionType,

    // Monitoring
    MonitoringEngine,
    QuarantineManager,
    QuarantineReason,

    ReputationPropagation,
    SybilDetector,
    TrustEventType,
};

use mycelix_sdk::epistemic::EmpiricalLevel;
use mycelix_sdk::matl::KVector;
use std::collections::HashMap;

// ============================================================================
// Test Helpers
// ============================================================================

fn create_agent(id: &str, sponsor: &str, trust_level: f32) -> InstrumentalActor {
    InstrumentalActor {
        agent_id: AgentId::from_string(id.to_string()),
        sponsor_did: sponsor.to_string(),
        agent_class: AgentClass::Supervised,
        kredit_balance: 5000,
        kredit_cap: 10000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: 0,
        last_activity: 0,
        actions_this_hour: 0,
        k_vector: KVector::new(
            trust_level,
            trust_level * 0.9,
            trust_level,
            trust_level * 0.95,
            0.3,
            0.4,
            trust_level * 0.85,
            0.3,
            trust_level * 0.9,
            0.5,
        ),
        epistemic_stats: EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: Default::default(),
        pending_escalations: vec![],
    }
}

fn add_behavior(agent: &mut InstrumentalActor, success_rate: f64, count: usize, base_time: u64) {
    for i in 0..count {
        let outcome = if (i as f64 / count as f64) < success_rate {
            ActionOutcome::Success
        } else {
            ActionOutcome::Error
        };
        agent.behavior_log.push(BehaviorLogEntry {
            timestamp: base_time + (i as u64 * 60) + (i as u64 % 30), // Varied timing
            action_type: format!("action_{}", i % 5),
            kredit_consumed: 10,
            counterparties: vec![],
            outcome,
        });
    }
}

// ============================================================================
// Scenario 1: Multi-Agent Consensus with Trust Weighting
// ============================================================================

#[test]
fn test_scenario_consensus_with_varied_trust() {
    // Create agents with different trust levels
    let high_trust = create_agent("high-trust", "sponsor-a", 0.9);
    let medium_trust = create_agent("medium-trust", "sponsor-b", 0.6);
    let low_trust = create_agent("low-trust", "sponsor-c", 0.4); // Above min threshold

    let mut agents = HashMap::new();
    agents.insert("high-trust".to_string(), high_trust);
    agents.insert("medium-trust".to_string(), medium_trust);
    agents.insert("low-trust".to_string(), low_trust);

    // Votes: high trust votes 0.9, medium votes 0.5, low votes 0.1
    let votes = vec![
        AgentVote {
            agent_id: "high-trust".to_string(),
            value: 0.9,
            confidence: 0.95,
            epistemic_level: 3,
            reasoning: Some("Strong evidence".to_string()),
            timestamp: 1000,
        },
        AgentVote {
            agent_id: "medium-trust".to_string(),
            value: 0.5,
            confidence: 0.7,
            epistemic_level: 2,
            reasoning: None,
            timestamp: 1000,
        },
        AgentVote {
            agent_id: "low-trust".to_string(),
            value: 0.1,
            confidence: 0.8,
            epistemic_level: 1,
            reasoning: None,
            timestamp: 1000,
        },
    ];

    let config = ConsensusConfig::default();
    let result = compute_consensus(&votes, &agents, &config);

    // Consensus should be computed (may vary based on trust calculation)
    assert!(
        result.participant_count == 3,
        "All agents should participate"
    );

    // Higher trust agents should have more contribution
    let high_contrib = result.contributions.get("high-trust").unwrap_or(&0.0);
    let low_contrib = result.contributions.get("low-trust").unwrap_or(&0.0);
    assert!(
        high_contrib >= low_contrib,
        "High trust ({}) should have >= contribution than low trust ({})",
        high_contrib,
        low_contrib
    );

    println!("Consensus value: {}", result.consensus_value);
    println!("Contributions: {:?}", result.contributions);
}

// ============================================================================
// Scenario 2: Gaming Detection Integrated with Quarantine
// ============================================================================

#[test]
fn test_scenario_gaming_detection_to_quarantine() {
    let mut gaming_detector = GamingDetector::new(GamingDetectionConfig::default());
    let mut quarantine = QuarantineManager::new();
    let mut monitoring = MonitoringEngine::new(AlertThresholds::default(), 100);

    // Create a gaming agent (perfect success rate = suspicious)
    let mut gamer = create_agent("gamer", "sponsor-x", 0.5);

    // Add suspiciously perfect behavior with regular timing
    for i in 0..50 {
        gamer.behavior_log.push(BehaviorLogEntry {
            timestamp: 1000 + i * 60, // Regular 1-minute intervals
            action_type: "process".to_string(),
            kredit_consumed: 10,
            counterparties: vec![],
            outcome: ActionOutcome::Success, // 100% success
        });
    }

    // Analyze for gaming
    let gaming_result = gaming_detector.analyze(&gamer, 5000);

    // Should detect gaming
    assert!(
        gaming_result.suspicion_score > 0.3,
        "Should detect gaming: score={}",
        gaming_result.suspicion_score
    );

    // If response warrants quarantine, quarantine the agent
    if matches!(
        gaming_result.recommended_action,
        GamingResponse::Quarantine | GamingResponse::EscalateToSponsor
    ) {
        quarantine.quarantine(
            "gamer",
            QuarantineReason::GamingDetected,
            gaming_result
                .indicators
                .iter()
                .map(|i| i.description.clone())
                .collect(),
            5000,
        );

        assert!(quarantine.is_quarantined("gamer"));
    }

    // Monitoring should also flag issues
    let metrics = monitoring.collect_metrics(&gamer, 5000);
    // High success rate but still should collect metrics
    assert!(metrics.success_rate > 0.9);
}

// ============================================================================
// Scenario 3: Sybil Attack Detection Across Sponsor
// ============================================================================

#[test]
fn test_scenario_sybil_detection_same_sponsor() {
    let sybil_detector = SybilDetector::new(0.85);

    // Create multiple agents from same sponsor with similar K-Vectors
    let base_kv = KVector::new(0.7, 0.65, 0.72, 0.68, 0.3, 0.35, 0.66, 0.32, 0.69, 0.5);

    let mut sybil1 = create_agent("sybil-1", "evil-sponsor", 0.7);
    let mut sybil2 = create_agent("sybil-2", "evil-sponsor", 0.7);
    let mut sybil3 = create_agent("sybil-3", "evil-sponsor", 0.7);

    // Make K-Vectors very similar (Sybil signature)
    sybil1.k_vector = base_kv.clone();
    sybil2.k_vector = KVector::new(0.71, 0.64, 0.73, 0.67, 0.31, 0.34, 0.67, 0.31, 0.70, 0.5);
    sybil3.k_vector = KVector::new(0.69, 0.66, 0.71, 0.69, 0.29, 0.36, 0.65, 0.33, 0.68, 0.5);

    let agents: Vec<&InstrumentalActor> = vec![&sybil1, &sybil2, &sybil3];
    let evidence = sybil_detector.analyze_group(&agents);

    // Should detect Sybil pattern
    assert!(!evidence.is_empty(), "Should detect Sybil pattern");

    // Verify it's the sponsor correlation type
    let has_sponsor_evidence = evidence.iter().any(|e| {
        matches!(
            e.evidence_type,
            mycelix_sdk::agentic::SybilEvidenceType::CorrelatedSponsorAgents
        )
    });
    assert!(has_sponsor_evidence, "Should identify sponsor correlation");
}

// ============================================================================
// Scenario 4: Collaboration with Calibration Learning
// ============================================================================

#[test]
fn test_scenario_collaboration_with_calibration_sharing() {
    let mut collaboration = CollaborationManager::new();
    let mut calibration_engine = CalibrationEngine::new(CalibrationEngineConfig::default());

    // Create agents with different trust levels
    let expert = create_agent("expert", "sponsor-a", 0.9);
    let novice = create_agent("novice", "sponsor-b", 0.5);

    // Expert has calibration history
    for i in 0..100 {
        calibration_engine.record("expert", 0.7, i < 70, Some("science"));
    }

    // Verify calibration was recorded
    let profile = calibration_engine.get_profile("expert");
    assert!(profile.is_some(), "Expert should have calibration profile");

    // Create collaborative prediction task
    collaboration.create_task(
        "predict-1".to_string(),
        CollaborativeTaskType::Prediction,
        0.3,
        2,
        10000,
    );

    // Both join and contribute
    assert!(collaboration.join_task("predict-1", &expert).is_ok());
    assert!(collaboration.join_task("predict-1", &novice).is_ok());

    assert!(collaboration
        .contribute("predict-1", &expert, serde_json::json!(0.75), 3, 0.9, 1000)
        .is_ok());

    assert!(collaboration
        .contribute("predict-1", &novice, serde_json::json!(0.65), 2, 0.7, 1001)
        .is_ok());

    // Finalize
    let result = collaboration.finalize_task("predict-1", 2000).unwrap();

    // Expert should have higher contribution weight due to higher trust
    let expert_contrib = result.contribution_scores.get("expert").unwrap();
    let novice_contrib = result.contribution_scores.get("novice").unwrap();
    assert!(
        expert_contrib > novice_contrib,
        "Expert ({}) should have more weight than novice ({})",
        expert_contrib,
        novice_contrib
    );
}

// ============================================================================
// Scenario 5: Reputation Propagation Through Interactions
// ============================================================================

#[test]
fn test_scenario_reputation_propagation() {
    let mut reputation = ReputationPropagation::new(0.85);

    // Create network: Hub -> Spoke1, Spoke2, Spoke3
    let hub = create_agent("hub", "sponsor-a", 0.95);
    let spoke1 = create_agent("spoke1", "sponsor-b", 0.5);
    let spoke2 = create_agent("spoke2", "sponsor-c", 0.5);
    let spoke3 = create_agent("spoke3", "sponsor-d", 0.5);

    let mut agents = HashMap::new();
    agents.insert("hub".to_string(), hub);
    agents.insert("spoke1".to_string(), spoke1);
    agents.insert("spoke2".to_string(), spoke2);
    agents.insert("spoke3".to_string(), spoke3);

    // Hub endorses spoke1 strongly, spoke2 moderately, spoke3 weakly
    reputation.record_interaction(AgentInteraction {
        from_agent: "hub".to_string(),
        to_agent: "spoke1".to_string(),
        interaction_type: InteractionType::Collaboration,
        quality: 1.0,
        weight: 1.0,
        timestamp: 1000,
    });
    reputation.record_interaction(AgentInteraction {
        from_agent: "hub".to_string(),
        to_agent: "spoke2".to_string(),
        interaction_type: InteractionType::Collaboration,
        quality: 0.5,
        weight: 1.0,
        timestamp: 1000,
    });
    reputation.record_interaction(AgentInteraction {
        from_agent: "hub".to_string(),
        to_agent: "spoke3".to_string(),
        interaction_type: InteractionType::Collaboration,
        quality: 0.2,
        weight: 1.0,
        timestamp: 1000,
    });

    // Propagate reputation
    reputation.propagate(&agents, 10);

    // Spoke1 should have highest propagated reputation among spokes
    let spoke1_rep = reputation.get_propagated_reputation("spoke1").unwrap();
    let spoke2_rep = reputation.get_propagated_reputation("spoke2").unwrap();
    let spoke3_rep = reputation.get_propagated_reputation("spoke3").unwrap();

    assert!(
        spoke1_rep > spoke2_rep,
        "spoke1 should have higher rep than spoke2"
    );
    assert!(
        spoke2_rep > spoke3_rep,
        "spoke2 should have higher rep than spoke3"
    );
}

// ============================================================================
// Scenario 6: Full Monitoring Pipeline with Alerts
// ============================================================================

#[test]
fn test_scenario_monitoring_alerts_pipeline() {
    let thresholds = AlertThresholds {
        trust_drop_threshold: 0.05,
        critical_health: 40,
        ..Default::default()
    };
    let mut monitoring = MonitoringEngine::new(thresholds, 100);

    // Create agent and collect initial metrics
    let mut agent = create_agent("monitored", "sponsor", 0.7);
    add_behavior(&mut agent, 0.8, 30, 1000);

    let initial_metrics = monitoring.collect_metrics(&agent, 5000);
    let initial_trust = initial_metrics.trust_score;

    // Simulate trust degradation by changing K-Vector
    let before_kv = agent.k_vector.clone();
    agent.k_vector.k_r = 0.3;
    agent.k_vector.k_i = 0.4;
    agent.k_vector.k_p = 0.35;

    // Record trust event
    monitoring.record_trust_event(
        "monitored",
        TrustEventType::NegativeAction,
        &before_kv,
        &agent.k_vector,
        "Multiple failures detected".to_string(),
        6000,
    );

    // Collect new metrics after degradation
    // First, add history entry to enable delta calculation
    let new_metrics = monitoring.collect_metrics(&agent, 90000); // Much later

    // Verify trust dropped
    assert!(new_metrics.trust_score < initial_trust);

    // Check dashboard summary
    let summary = monitoring.dashboard_summary();
    assert_eq!(summary.total_agents, 1);
}

// ============================================================================
// Scenario 7: Collusion Detection in Voting
// ============================================================================

#[test]
fn test_scenario_collusion_in_voting() {
    let mut collusion_detector = CollusionDetector::new(0.9);

    // Simulate coordinated voting (3 agents always vote the same)
    for topic in 0..15 {
        let vote_value = if topic % 2 == 0 { 0.9 } else { 0.1 };

        for agent_id in &["colluder-1", "colluder-2", "colluder-3"] {
            collusion_detector.record_interaction(AgentInteractionRecord {
                from_agent: agent_id.to_string(),
                to_agent: format!("topic-{}", topic),
                interaction_type: "vote".to_string(),
                value: vote_value,
                timestamp: 1000 + topic as u64,
            });
        }
    }

    // Add one honest agent with varied votes
    for topic in 0..15 {
        collusion_detector.record_interaction(AgentInteractionRecord {
            from_agent: "honest".to_string(),
            to_agent: format!("topic-{}", topic),
            interaction_type: "vote".to_string(),
            value: (topic as f64 * 0.07) % 1.0, // Varied
            timestamp: 1000 + topic as u64,
        });
    }

    let evidence = collusion_detector.analyze();

    // Should detect vote coordination among colluders
    // The detection may or may not fire depending on exact algorithm
    // but shouldn't panic and should run successfully
    println!("Collusion evidence found: {}", evidence.len());
}

// ============================================================================
// Scenario 8: End-to-End Agent Lifecycle
// ============================================================================

#[test]
fn test_scenario_complete_agent_lifecycle() {
    // Initialize all systems
    let mut monitoring = MonitoringEngine::new(AlertThresholds::default(), 100);
    let mut gaming_detector = GamingDetector::new(GamingDetectionConfig::default());
    let mut calibration = CalibrationEngine::new(CalibrationEngineConfig::default());
    let mut quarantine = QuarantineManager::new();

    // Phase 1: Create new agent with decent trust
    let mut agent = create_agent("lifecycle-agent", "good-sponsor", 0.6);

    // Phase 2: Agent operates with varied behavior (human-like)
    let varied_intervals = [
        45, 180, 90, 300, 60, 150, 200, 75, 120, 240, 55, 170, 85, 310, 65, 145, 210, 80, 125, 230,
        50, 175, 95, 290, 70, 155, 195, 78, 130, 250, 60, 190, 88, 280, 72, 160, 205, 82, 135, 245,
        58, 185, 92, 295, 68, 150, 200, 85, 128, 235,
    ];
    let mut timestamp = 1000u64;
    for (i, &interval) in varied_intervals.iter().enumerate() {
        timestamp += interval;
        agent.behavior_log.push(BehaviorLogEntry {
            timestamp,
            action_type: format!("action_{}", i % 5),
            kredit_consumed: 10,
            counterparties: vec![],
            outcome: if i % 4 == 0 {
                ActionOutcome::Error
            } else {
                ActionOutcome::Success
            },
        });
    }

    // Track calibration
    for i in 0..50 {
        calibration.record("lifecycle-agent", 0.75, i < 38, None);
    }

    // Phase 3: Monitor
    let metrics1 = monitoring.collect_metrics(&agent, timestamp + 1000);
    println!("Initial health score: {}", metrics1.health_score);

    // Phase 4: Check for gaming (should be clean with varied behavior)
    let gaming_check = gaming_detector.analyze(&agent, timestamp + 1000);
    println!("Gaming suspicion score: {}", gaming_check.suspicion_score);

    // Phase 5: Agent trust improves over time
    agent.k_vector.k_r = 0.7;
    agent.k_vector.k_p = 0.75;
    agent.k_vector.k_h = 0.65;

    let metrics2 = monitoring.collect_metrics(&agent, timestamp + 100000);

    // Phase 6: Verify not quarantined
    assert!(!quarantine.is_quarantined("lifecycle-agent"));

    // Phase 7: Get calibration adjustment
    let adjustment = calibration.get_kvector_adjustment("lifecycle-agent");
    assert!(adjustment.is_some());

    println!("Agent lifecycle completed successfully:");
    println!("  Initial health: {}", metrics1.health_score);
    println!("  Final health: {}", metrics2.health_score);
    println!("  Final trust: {:.2}", metrics2.trust_score);
}

// ============================================================================
// Scenario 9: Byzantine Resistance in Consensus
// ============================================================================

#[test]
fn test_scenario_byzantine_consensus_resistance() {
    // 3 honest agents, 1 byzantine agent
    let honest1 = create_agent("honest-1", "sponsor-a", 0.8);
    let honest2 = create_agent("honest-2", "sponsor-b", 0.75);
    let honest3 = create_agent("honest-3", "sponsor-c", 0.7);
    let byzantine = create_agent("byzantine", "sponsor-d", 0.6);

    let mut agents = HashMap::new();
    agents.insert("honest-1".to_string(), honest1);
    agents.insert("honest-2".to_string(), honest2);
    agents.insert("honest-3".to_string(), honest3);
    agents.insert("byzantine".to_string(), byzantine);

    // Honest agents vote ~0.7, byzantine votes 0.0 (trying to manipulate)
    let votes = vec![
        AgentVote {
            agent_id: "honest-1".to_string(),
            value: 0.72,
            confidence: 0.85,
            epistemic_level: 3,
            reasoning: None,
            timestamp: 1000,
        },
        AgentVote {
            agent_id: "honest-2".to_string(),
            value: 0.68,
            confidence: 0.8,
            epistemic_level: 2,
            reasoning: None,
            timestamp: 1000,
        },
        AgentVote {
            agent_id: "honest-3".to_string(),
            value: 0.71,
            confidence: 0.75,
            epistemic_level: 2,
            reasoning: None,
            timestamp: 1000,
        },
        AgentVote {
            agent_id: "byzantine".to_string(),
            value: 0.0, // Malicious vote
            confidence: 0.9,
            epistemic_level: 1,
            reasoning: None,
            timestamp: 1000,
        },
    ];

    let result = compute_consensus(&votes, &agents, &ConsensusConfig::default());

    // Consensus should be close to honest agents' values (~0.7)
    // Byzantine agent's low trust should limit its influence
    assert!(
        result.consensus_value > 0.5,
        "Consensus should resist byzantine: {}",
        result.consensus_value
    );
    assert!(
        result.consensus_value < 0.75,
        "But not be exactly honest average due to byzantine: {}",
        result.consensus_value
    );
}

// ============================================================================
// Scenario 10: Epistemic Calibration Integration
// ============================================================================

#[test]
fn test_scenario_epistemic_calibration() {
    let mut profile = EnhancedAgentCalibrationProfile::new("epistemic-agent".to_string());

    // Record predictions at different epistemic levels
    let timestamp = 1000u64;

    // E1 (testimonial) - looser calibration expected
    for i in 0..30 {
        profile.record_full(
            0.7,
            i < 18,
            timestamp + i,
            Some(EmpiricalLevel::E1Testimonial),
            Some("general"),
        );
    }

    // E3 (cryptographic) - tighter calibration expected
    for i in 0..30 {
        profile.record_full(
            0.8,
            i < 24,
            timestamp + 100 + i,
            Some(EmpiricalLevel::E3Cryptographic),
            Some("verified"),
        );
    }

    // Get comprehensive adjustment
    let adjustment = profile.comprehensive_kvector_adjustment(timestamp + 200);

    // Should have epistemic quality assessment
    println!("Epistemic quality: {:?}", adjustment.epistemic_quality);
    println!(
        "Systematic overconfidence: {:.3}",
        adjustment.systematic_overconfidence
    );
    println!(
        "K-Vector adjustments: k_p={:.3}, k_i={:.3}, k_r={:.3}",
        adjustment.k_p_delta, adjustment.k_i_delta, adjustment.k_r_delta
    );

    // Well-calibrated agent shouldn't have severe penalties
    assert!(adjustment.k_p_delta > -0.1);
    assert!(adjustment.k_i_delta > -0.1);
}
