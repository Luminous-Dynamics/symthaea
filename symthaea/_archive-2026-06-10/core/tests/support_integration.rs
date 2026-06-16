#![cfg(feature = "support")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Integration tests for the symthaea-support crate.
//!
//! These tests exercise cross-module workflows: triage → knowledge,
//! diagnostics → actions, scrubber → cognitive updates, predictive alerts,
//! autonomy gates, knowledge graduation, and privacy tier controls.

use symthaea_support::actions::ActionEngine;
use symthaea_support::diagnostics::DiagnosticPlanner;
use symthaea_support::knowledge::KnowledgeManager;
use symthaea_support::predictive::PredictiveEngine;
use symthaea_support::privacy::{PrivacyManager, SharingTier};
use symthaea_support::scrubber::PrivacyScrubber;
use symthaea_support::triage::TriageEngine;
use symthaea_support::types::*;

// ---------------------------------------------------------------------------
// 1. End-to-end triage -> knowledge search
// ---------------------------------------------------------------------------

#[test]
fn triage_then_knowledge_search() {
    // Triage a network ticket
    let triage = TriageEngine::new();
    let result = triage.triage(
        "DNS resolution fails intermittently",
        "Our holochain conductor loses connectivity when DNS goes down on the network",
    );

    // The triage engine should classify this into a category
    assert!(
        result.suggested_category == SupportCategory::Network
            || result.suggested_category == SupportCategory::Holochain,
        "Expected Network or Holochain, got {:?}",
        result.suggested_category
    );

    // Now use the triage category to search the knowledge base
    let mut km = KnowledgeManager::new();
    km.add_article(
        "KB-001".to_string(),
        "DNS Troubleshooting Guide".to_string(),
        SupportCategory::Network,
        0.8,
    );
    km.add_article(
        "KB-002".to_string(),
        "Holochain Conductor Restart Procedure".to_string(),
        SupportCategory::Holochain,
        0.7,
    );
    km.add_article(
        "KB-003".to_string(),
        "DNS Cache Flush Steps".to_string(),
        SupportCategory::Network,
        0.6,
    );

    // Search by a keyword derived from the triage result
    let matches = km.search("DNS", 10);
    assert!(
        matches.len() >= 2,
        "Expected at least 2 DNS articles, got {}",
        matches.len()
    );
    // Results should be sorted by phi descending
    assert!(matches[0].phi >= matches[1].phi);
}

// ---------------------------------------------------------------------------
// 2. Knowledge lifecycle: add -> search -> verify ranking -> deprecate -> verify exclusion
// ---------------------------------------------------------------------------

#[test]
fn knowledge_lifecycle() {
    let mut km = KnowledgeManager::new();

    // Seed articles with varying phi
    km.add_article(
        "A1".to_string(),
        "Firewall Configuration".to_string(),
        SupportCategory::Security,
        0.9,
    );
    km.add_article(
        "A2".to_string(),
        "Firewall Logs Analysis".to_string(),
        SupportCategory::Security,
        0.4,
    );
    km.add_article(
        "A3".to_string(),
        "Firewall Rules Update".to_string(),
        SupportCategory::Security,
        0.7,
    );

    // Search and verify phi-ranked ordering
    let results = km.search("Firewall", 10);
    assert_eq!(results.len(), 3);
    assert!(
        (results[0].phi - 0.9).abs() < f64::EPSILON,
        "First should be phi=0.9"
    );
    assert!(
        (results[1].phi - 0.7).abs() < f64::EPSILON,
        "Second should be phi=0.7"
    );
    assert!(
        (results[2].phi - 0.4).abs() < f64::EPSILON,
        "Third should be phi=0.4"
    );

    // Deprecate the top article
    let deprecated = km.deprecate("A1");
    assert!(
        deprecated,
        "Deprecation should succeed for existing article"
    );

    // Verify it is excluded from future searches
    let after = km.search("Firewall", 10);
    assert_eq!(after.len(), 2, "Deprecated article should be excluded");
    assert_eq!(
        after[0].article_id, "A3",
        "Highest remaining phi should be first"
    );
    assert_eq!(after[1].article_id, "A2");

    // Deprecating a non-existent article returns false
    assert!(!km.deprecate("DOES_NOT_EXIST"));
}

// ---------------------------------------------------------------------------
// 3. Diagnostic plan -> action proposal
// ---------------------------------------------------------------------------

#[test]
fn diagnostic_plan_then_action_proposal() {
    let planner = DiagnosticPlanner::new();
    let symptoms = vec![
        "holochain conductor not responding".to_string(),
        "network timeout on dht sync".to_string(),
    ];
    let plan = planner.plan_diagnostics(&symptoms);

    // Plan should include HolochainHealth and NetworkCheck steps
    let has_holochain = plan
        .steps
        .iter()
        .any(|s| s.diagnostic_type == DiagnosticType::HolochainHealth);
    let has_network = plan
        .steps
        .iter()
        .any(|s| s.diagnostic_type == DiagnosticType::NetworkCheck);
    assert!(has_holochain, "Should include HolochainHealth diagnostic");
    assert!(has_network, "Should include NetworkCheck diagnostic");

    // Steps should be sorted by expected info gain descending
    for i in 0..plan.steps.len() - 1 {
        assert!(
            plan.steps[i].expected_info_gain >= plan.steps[i + 1].expected_info_gain,
            "Steps not sorted by info gain at index {}",
            i
        );
    }

    // Based on diagnostic findings, propose an action
    let action_engine = ActionEngine::new();

    // For each autonomous-capable diagnostic step, propose an appropriate action
    let autonomous_steps: Vec<_> = plan.steps.iter().filter(|s| s.autonomous_capable).collect();
    assert!(
        !autonomous_steps.is_empty(),
        "At least some steps should be autonomous-capable"
    );

    let proposed = action_engine.propose_action(
        ActionType::RestartService,
        "Restart conductor based on HolochainHealth diagnostic".to_string(),
    );
    assert_eq!(proposed.action_type, ActionType::RestartService);
    assert!(
        !proposed.rollback_steps.is_empty(),
        "Proposed action should have rollback steps"
    );
    assert!(proposed.confidence > 0.0);
}

// ---------------------------------------------------------------------------
// 4. Privacy scrubber -> cognitive update roundtrip
// ---------------------------------------------------------------------------

#[test]
fn scrubber_then_cognitive_update() {
    let scrubber = PrivacyScrubber::new();

    // Scrub PII from a support description
    let raw_text = "User admin@mycelix.net reported that server 10.0.0.42 at \
                     /home/tstoltz/.config/holochain is failing with key sk-ABCDEFGHIJKLMNOPQR";
    let scrub_result = scrubber.scrub(raw_text);

    assert!(
        scrub_result.redaction_count >= 4,
        "Should redact IP, email, path, and key"
    );
    assert!(!scrub_result.scrubbed_text.contains("admin@mycelix.net"));
    assert!(!scrub_result.scrubbed_text.contains("10.0.0.42"));
    assert!(!scrub_result.scrubbed_text.contains("/home/tstoltz"));
    assert!(!scrub_result.scrubbed_text.contains("sk-ABCDEFGHIJKLMNOPQR"));

    // Use scrubbed text in a cognitive update
    let update = CognitiveUpdateData {
        category: SupportCategory::Holochain,
        encoding: scrub_result.scrubbed_text.as_bytes().to_vec(),
        phi: 0.65,
        resolution_pattern: scrub_result.scrubbed_text.clone(),
    };

    // Verify the cognitive update contains no PII
    assert!(!update.resolution_pattern.contains("admin@mycelix.net"));
    assert!(!update.resolution_pattern.contains("10.0.0.42"));
    assert!(update.resolution_pattern.contains("[REDACTED_"));

    // Feed into knowledge manager
    let mut km = KnowledgeManager::new();
    let absorption = km.absorb_cognitive_update(&update);
    assert!(absorption.absorbed, "Cognitive update should be absorbed");
}

// ---------------------------------------------------------------------------
// 5. Predictive engine -> alert workflow
// ---------------------------------------------------------------------------

#[test]
fn predictive_alert_workflow() {
    let engine = PredictiveEngine::new();

    // Inject unhealthy telemetry (multiple issues)
    let unhealthy = SystemTelemetry {
        conductor_memory_mb: 8192.0,
        gossip_peers: 0,
        disk_free_pct: 3.0,
        network_latency_ms: 1200.0,
        error_rate_per_min: 25.0,
    };
    let prediction = engine.assess_system_state(&unhealthy);

    assert!(
        prediction.expected_free_energy > 5.0,
        "Unhealthy system should have high free energy, got {}",
        prediction.expected_free_energy
    );
    assert!(
        prediction.predicted_failure.is_some(),
        "Should predict failure for unhealthy system"
    );
    assert!(
        engine.should_alert(&prediction),
        "Should trigger alert for unhealthy system"
    );
    assert_eq!(
        prediction.time_horizon_secs,
        Some(3600),
        "Should set time horizon when above threshold"
    );

    // Inject healthy telemetry
    let healthy = SystemTelemetry {
        conductor_memory_mb: 512.0,
        gossip_peers: 8,
        disk_free_pct: 75.0,
        network_latency_ms: 10.0,
        error_rate_per_min: 0.05,
    };
    let healthy_prediction = engine.assess_system_state(&healthy);

    assert!(
        healthy_prediction.expected_free_energy < 1.0,
        "Healthy system should have low free energy, got {}",
        healthy_prediction.expected_free_energy
    );
    assert!(
        healthy_prediction.predicted_failure.is_none(),
        "Should not predict failure for healthy system"
    );
    assert!(
        !engine.should_alert(&healthy_prediction),
        "Should not alert for healthy system"
    );
    assert_eq!(
        healthy_prediction.time_horizon_secs, None,
        "No time horizon for healthy system"
    );
}

// ---------------------------------------------------------------------------
// 6. Action engine autonomy gates
// ---------------------------------------------------------------------------

#[test]
fn action_engine_autonomy_gates() {
    let engine = ActionEngine::new();

    let actions = vec![
        ActionType::RestartService,
        ActionType::ClearCache,
        ActionType::UpdateConfig,
        ActionType::RunDiagnostic,
        ActionType::Custom("deploy-hotfix".to_string()),
    ];

    // Advisory: blocks everything
    for action in &actions {
        assert!(
            !engine.can_execute(action, &AutonomyLevel::Advisory),
            "Advisory should block {:?}",
            action
        );
    }

    // SemiAutonomous: allows only permitted actions (RestartService, ClearCache, RunDiagnostic)
    assert!(engine.can_execute(&ActionType::RestartService, &AutonomyLevel::SemiAutonomous));
    assert!(engine.can_execute(&ActionType::ClearCache, &AutonomyLevel::SemiAutonomous));
    assert!(engine.can_execute(&ActionType::RunDiagnostic, &AutonomyLevel::SemiAutonomous));
    assert!(
        !engine.can_execute(&ActionType::UpdateConfig, &AutonomyLevel::SemiAutonomous),
        "SemiAutonomous should block UpdateConfig (not in permitted list)"
    );
    assert!(
        !engine.can_execute(
            &ActionType::Custom("deploy-hotfix".to_string()),
            &AutonomyLevel::SemiAutonomous
        ),
        "SemiAutonomous should block Custom actions"
    );

    // FullAutonomous: allows all
    for action in &actions {
        assert!(
            engine.can_execute(action, &AutonomyLevel::FullAutonomous),
            "FullAutonomous should allow {:?}",
            action
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Knowledge graduation workflow
// ---------------------------------------------------------------------------

#[test]
fn knowledge_graduation_thresholds() {
    let km = KnowledgeManager::new();

    // Graduate: high phi + high effectiveness + sufficient retrievals
    let graduate = km.evaluate_for_graduation(0.7, 5, 5);
    assert!(
        matches!(graduate, GraduationDecision::Graduate),
        "High quality should graduate, got {:?}",
        graduate
    );

    // Also graduate at the boundary values
    let boundary = km.evaluate_for_graduation(0.51, 4, 3);
    assert!(
        matches!(boundary, GraduationDecision::Graduate),
        "Boundary values should graduate, got {:?}",
        boundary
    );

    // Defer: moderate phi (>0.3) but insufficient effectiveness or retrievals
    let defer_phi = km.evaluate_for_graduation(0.4, 2, 1);
    assert!(
        matches!(defer_phi, GraduationDecision::Defer(_)),
        "Moderate phi with low effectiveness should defer, got {:?}",
        defer_phi
    );

    // Defer: low phi but sufficient retrievals (>=2)
    let defer_retrieval = km.evaluate_for_graduation(0.1, 1, 2);
    assert!(
        matches!(defer_retrieval, GraduationDecision::Defer(_)),
        "Low phi but sufficient retrievals should defer, got {:?}",
        defer_retrieval
    );

    // Reject: low phi + low effectiveness + low retrievals
    let reject = km.evaluate_for_graduation(0.1, 1, 0);
    assert!(
        matches!(reject, GraduationDecision::Reject(_)),
        "Low quality should reject, got {:?}",
        reject
    );

    let reject_2 = km.evaluate_for_graduation(0.2, 2, 1);
    assert!(
        matches!(reject_2, GraduationDecision::Reject(_)),
        "Below all thresholds should reject, got {:?}",
        reject_2
    );
}

// ---------------------------------------------------------------------------
// 8. Cognitive update absorption
// ---------------------------------------------------------------------------

#[test]
fn cognitive_update_absorption() {
    let mut km = KnowledgeManager::new();

    // Create a cognitive update for a category with no existing articles
    let update_new = CognitiveUpdateData {
        category: SupportCategory::Security,
        encoding: vec![42, 43, 44, 45],
        phi: 0.72,
        resolution_pattern: "Rotate certificates and restart TLS handshake".to_string(),
    };

    let result_new = km.absorb_cognitive_update(&update_new);
    assert!(result_new.absorbed);
    assert!(
        (result_new.similarity_to_existing - 0.0).abs() < f32::EPSILON,
        "No existing articles in category, similarity should be 0.0"
    );
    assert!(result_new.message.contains("Security"));
    assert!(result_new.message.contains("0.720"));

    // Now add an article in the same category and absorb again
    km.add_article(
        "SEC-001".to_string(),
        "Certificate Rotation Guide".to_string(),
        SupportCategory::Security,
        0.6,
    );

    let update_existing = CognitiveUpdateData {
        category: SupportCategory::Security,
        encoding: vec![50, 51, 52],
        phi: 0.55,
        resolution_pattern: "Apply security patch".to_string(),
    };

    let result_existing = km.absorb_cognitive_update(&update_existing);
    assert!(result_existing.absorbed);
    assert!(
        result_existing.similarity_to_existing > 0.0,
        "Should detect existing articles in same category"
    );
    assert!(
        (result_existing.similarity_to_existing - 0.5).abs() < f32::EPSILON,
        "Expected 0.5 similarity when category match exists"
    );
}

// ---------------------------------------------------------------------------
// 9. Scrubber handles complex mixed PII
// ---------------------------------------------------------------------------

#[test]
fn scrubber_complex_mixed_pii() {
    let scrubber = PrivacyScrubber::new();

    let complex_text = "Ticket from user@example.com and admin@mycelix.net: \
                        Server 192.168.1.100 and 10.0.0.1 have issues. \
                        Logs at /home/tstoltz/logs/conductor.log and /home/admin/.config/holochain. \
                        Using API key api-ABCDEFGHIJKLMNOPQ and token token-1234567890abcdefgh for auth.";

    let result = scrubber.scrub(complex_text);

    // Verify all PII types are redacted
    assert!(
        !result.scrubbed_text.contains("user@example.com"),
        "Email 1 not redacted"
    );
    assert!(
        !result.scrubbed_text.contains("admin@mycelix.net"),
        "Email 2 not redacted"
    );
    assert!(
        !result.scrubbed_text.contains("192.168.1.100"),
        "IP 1 not redacted"
    );
    assert!(
        !result.scrubbed_text.contains("10.0.0.1"),
        "IP 2 not redacted"
    );
    assert!(
        !result.scrubbed_text.contains("/home/tstoltz"),
        "Path 1 not redacted"
    );
    assert!(
        !result.scrubbed_text.contains("/home/admin"),
        "Path 2 not redacted"
    );
    assert!(
        !result.scrubbed_text.contains("api-ABCDEFGHIJKLMNOPQ"),
        "API key not redacted"
    );
    assert!(
        !result.scrubbed_text.contains("token-1234567890abcdefgh"),
        "Token not redacted"
    );

    // Verify redaction counts by type
    assert_eq!(
        result.redaction_types.get("email").copied().unwrap_or(0),
        2,
        "Should have 2 emails"
    );
    assert!(
        result.redaction_types.get("ip").copied().unwrap_or(0) >= 2,
        "Should have at least 2 IPs"
    );
    assert_eq!(
        result.redaction_types.get("path").copied().unwrap_or(0),
        2,
        "Should have 2 paths"
    );
    assert_eq!(
        result.redaction_types.get("key").copied().unwrap_or(0),
        2,
        "Should have 2 keys"
    );

    // Total redaction count should cover all
    assert!(
        result.redaction_count >= 8,
        "Should have at least 8 redactions, got {}",
        result.redaction_count
    );

    // Verify that non-PII content is preserved
    assert!(result.scrubbed_text.contains("Ticket from"));
    assert!(result.scrubbed_text.contains("have issues"));
    assert!(result.scrubbed_text.contains("for auth"));
}

// ---------------------------------------------------------------------------
// 10. Privacy tier controls sharing
// ---------------------------------------------------------------------------

#[test]
fn privacy_tier_controls_sharing() {
    // LocalOnly: blocks all sharing
    let local = PrivacyManager::new(SharingTier::LocalOnly);
    assert!(!local.can_share(), "LocalOnly should block can_share");
    assert!(
        !local.can_share_cognitive(),
        "LocalOnly should block can_share_cognitive"
    );

    // Anonymized: allows sharing (both general and cognitive)
    let anon = PrivacyManager::new(SharingTier::Anonymized);
    assert!(anon.can_share(), "Anonymized should allow can_share");
    assert!(
        anon.can_share_cognitive(),
        "Anonymized should allow can_share_cognitive"
    );

    // Full: allows everything
    let full = PrivacyManager::new(SharingTier::Full);
    assert!(full.can_share(), "Full should allow can_share");
    assert!(
        full.can_share_cognitive(),
        "Full should allow can_share_cognitive"
    );

    // Verify tier transitions
    let mut pm = PrivacyManager::new(SharingTier::LocalOnly);
    assert!(!pm.can_share());

    pm.set_sharing_tier(SharingTier::Anonymized);
    assert!(pm.can_share());
    assert!(pm.can_share_cognitive());

    pm.set_sharing_tier(SharingTier::Full);
    assert!(pm.can_share());
    assert!(pm.can_share_cognitive());

    // Downgrade back to LocalOnly
    pm.set_sharing_tier(SharingTier::LocalOnly);
    assert!(!pm.can_share());
    assert!(!pm.can_share_cognitive());
}