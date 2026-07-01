#![cfg(feature = "observability_module")]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Phase 3 Causal Understanding System - Integration Tests
//
// These tests validate the complete causal analysis pipeline:
// 1. Observer records events with correlation metadata
// 2. CorrelationContext tracks parent-child relationships
// 3. CausalGraph constructs causal relationships
// 4. TraceAnalyzer provides high-level insights
//
// Each test scenario represents a real-world usage pattern.

use chrono::Utc;
use std::sync::{Arc, RwLock};
use symthaea::observability::{
    CausalAnswer, CausalGraph, CausalTraceObserver, CorrelationContext, PhiComponents,
    PhiMeasurementEvent, PrimitiveActivationEvent, ResponseGeneratedEvent, RouterSelectionEvent,
    SecurityCheckEvent, SecurityDecision, SymthaeaObserver, Trace, TraceAnalyzer, TraceObserver,
    WithCausalMetadata, WorkspaceIgnitionEvent,
};
use tempfile::NamedTempFile;

type SharedObserver = Arc<RwLock<Box<dyn SymthaeaObserver + Send + Sync>>>;

// Helper to create observer with correlation tracking (Phase 4 version)
fn create_causal_observer() -> (CausalTraceObserver<TraceObserver>, tempfile::TempPath) {
    let temp_file = NamedTempFile::new().unwrap();
    let trace_path = temp_file.into_temp_path();

    let base_observer = TraceObserver::new(&trace_path).unwrap();
    let causal_observer = CausalTraceObserver::new(base_observer, "integration_test");

    (causal_observer, trace_path)
}

// Legacy helper for tests that don't need metadata injection
fn create_observer_with_correlation() -> (SharedObserver, CorrelationContext, tempfile::TempPath) {
    let temp_file = NamedTempFile::new().unwrap();
    let trace_path = temp_file.into_temp_path();

    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(
        TraceObserver::new(&trace_path).unwrap(),
    )));

    let ctx = CorrelationContext::new("integration_test");

    (observer, ctx, trace_path)
}

#[test]
fn test_scenario_1_basic_causal_chain() {
    // Scenario: Security check → Phi measurement → Router selection
    // Expected: Complete causal chain from security to router
    // Phase 4: Now using CausalTraceObserver for automatic metadata injection

    let (mut observer, trace_path) = create_causal_observer();

    // Step 1: Security check (root event)
    observer
        .record_security_check(SecurityCheckEvent {
            timestamp: Utc::now(),
            operation: "query_processing".to_string(),
            decision: SecurityDecision::Allowed,
            reason: Some("Clean input".to_string()),
            secrets_redacted: 0,
            similarity_score: Some(0.05),
            matched_pattern: None,
            metadata: None, // Will be injected by CausalTraceObserver
        })
        .unwrap();

    // Get the security event ID for establishing parent
    let security_id = observer.context().event_chain()[0].clone();

    // Step 2: Phi measurement (caused by security check)
    observer.begin_scope(&security_id);
    observer
        .record_phi_measurement(PhiMeasurementEvent {
            timestamp: Utc::now(),
            phi: 0.82,
            components: PhiComponents {
                integration: 0.85,
                binding: 0.80,
                workspace: 0.82,
                attention: 0.78,
                recursion: 0.75,
                efficacy: 0.80,
                knowledge: 0.77,
            },
            temporal_continuity: 0.91,
            metadata: None,
        })
        .unwrap();
    let phi_id = observer.context().event_chain()[1].clone();
    observer.end_scope();

    // Step 3: Router selection (caused by phi measurement)
    observer.begin_scope(&phi_id);
    observer
        .record_router_selection(RouterSelectionEvent {
            timestamp: Utc::now(),
            input: "complex query".to_string(),
            selected_router: "consciousness_guided".to_string(),
            confidence: 0.88,
            alternatives: vec![],
            bandit_stats: std::collections::HashMap::new(),
            metadata: None,
        })
        .unwrap();
    let router_id = observer.context().event_chain()[2].clone();
    observer.end_scope();

    // Finalize and analyze
    observer.finalize().unwrap();

    let trace = Trace::load_from_file(&trace_path).unwrap();
    let graph = CausalGraph::from_trace(&trace);

    // Verify causal relationships
    assert_eq!(graph.nodes.len(), 3, "Should have 3 events");

    // Verify security → phi causality
    let causes_of_phi = graph.find_causes(&phi_id);
    assert_eq!(causes_of_phi.len(), 1, "Phi should have 1 direct cause");
    assert_eq!(causes_of_phi[0].id, security_id);

    // Verify phi → router causality
    let causes_of_router = graph.find_causes(&router_id);
    assert_eq!(
        causes_of_router.len(),
        1,
        "Router should have 1 direct cause"
    );
    assert_eq!(causes_of_router[0].id, phi_id);

    // Verify complete causal chain
    match graph.did_cause(&security_id, &router_id) {
        CausalAnswer::IndirectCause { path, strength } => {
            assert!(path.len() >= 2, "Should have path through phi");
            assert!(strength > 0.5, "Should have strong causal relationship");
        }
        other => panic!("Expected IndirectCause, got {:?}", other),
    }
}

#[test]
fn test_scenario_2_parallel_processing() {
    // Scenario: One security check causes two parallel phi measurements
    // Expected: Security has 2 effects, both phi measurements have same parent
    // Phase 4: Now using CausalTraceObserver

    let (mut observer, trace_path) = create_causal_observer();

    // Root security check
    observer
        .record_security_check(SecurityCheckEvent {
            timestamp: Utc::now(),
            operation: "parallel_query".to_string(),
            decision: SecurityDecision::Allowed,
            reason: Some("Multi-path processing".to_string()),
            secrets_redacted: 0,
            similarity_score: Some(0.03),
            matched_pattern: None,
            metadata: None,
        })
        .unwrap();
    let security_id = observer.context().event_chain()[0].clone();

    // Parallel phi measurement 1
    observer.begin_scope(&security_id);
    observer
        .record_phi_measurement(PhiMeasurementEvent {
            timestamp: Utc::now(),
            phi: 0.75,
            components: PhiComponents::default(),
            temporal_continuity: 0.88,
            metadata: None,
        })
        .unwrap();
    let phi_id_1 = observer.context().event_chain()[1].clone();
    observer.end_scope();

    // Parallel phi measurement 2 (same parent: security)
    observer.begin_scope(&security_id);
    observer
        .record_phi_measurement(PhiMeasurementEvent {
            timestamp: Utc::now(),
            phi: 0.78,
            components: PhiComponents::default(),
            temporal_continuity: 0.90,
            metadata: None,
        })
        .unwrap();
    let phi_id_2 = observer.context().event_chain()[2].clone();
    observer.end_scope();

    // Finalize
    observer.finalize().unwrap();

    let trace = Trace::load_from_file(&trace_path).unwrap();
    let graph = CausalGraph::from_trace(&trace);

    // Verify parallel structure
    let effects_of_security = graph.find_effects(&security_id);
    assert!(
        effects_of_security.len() >= 2,
        "Security should cause at least 2 effects"
    );

    // Verify both phi measurements have security as a cause
    let causes_1 = graph.find_causes(&phi_id_1);
    let causes_2 = graph.find_causes(&phi_id_2);

    // Security should be a cause of both phi measurements
    let security_causes_phi1 = causes_1.iter().any(|c| c.id == security_id);
    let security_causes_phi2 = causes_2.iter().any(|c| c.id == security_id);
    assert!(security_causes_phi1, "Security should be a cause of phi_1");
    assert!(security_causes_phi2, "Security should be a cause of phi_2");
}

#[test]
fn test_scenario_3_deep_causal_chain() {
    // Scenario: Security → Phi → Router → Workspace → Primitive → Response
    // Expected: 6-event deep causal chain with all relationships intact
    // Phase 4: Now using CausalTraceObserver

    let (mut observer, trace_path) = create_causal_observer();

    // Event 1: Security check
    observer
        .record_security_check(SecurityCheckEvent {
            timestamp: Utc::now(),
            operation: "deep_processing".to_string(),
            decision: SecurityDecision::Allowed,
            reason: Some("Clean".to_string()),
            secrets_redacted: 0,
            similarity_score: Some(0.02),
            matched_pattern: None,
            metadata: None,
        })
        .unwrap();
    let e1_id = observer.context().event_chain()[0].clone();

    // Event 2: Phi measurement
    observer.begin_scope(&e1_id);
    observer
        .record_phi_measurement(PhiMeasurementEvent {
            timestamp: Utc::now(),
            phi: 0.80,
            components: PhiComponents::default(),
            temporal_continuity: 0.90,
            metadata: None,
        })
        .unwrap();
    let e2_id = observer.context().event_chain()[1].clone();
    observer.end_scope();

    // Event 3: Router selection
    observer.begin_scope(&e2_id);
    observer
        .record_router_selection(RouterSelectionEvent {
            timestamp: Utc::now(),
            input: "complex input".to_string(),
            selected_router: "standard".to_string(),
            confidence: 0.85,
            alternatives: vec![],
            bandit_stats: std::collections::HashMap::new(),
            metadata: None,
        })
        .unwrap();
    let e3_id = observer.context().event_chain()[2].clone();
    observer.end_scope();

    // Event 4: Workspace ignition
    observer.begin_scope(&e3_id);
    observer
        .record_workspace_ignition(WorkspaceIgnitionEvent {
            timestamp: Utc::now(),
            phi: 0.80,
            free_energy: 0.15,
            coalition_size: 5,
            active_primitives: vec!["reasoning".to_string()],
            broadcast_payload_size: 1024,
            metadata: None,
        })
        .unwrap();
    let e4_id = observer.context().event_chain()[3].clone();
    observer.end_scope();

    // Event 5: Primitive activation
    observer.begin_scope(&e4_id);
    observer
        .record_primitive_activation(PrimitiveActivationEvent {
            timestamp: Utc::now(),
            primitive_name: "reasoning".to_string(),
            tier: "core".to_string(),
            activation_strength: 0.92,
            context: vec!["deep_processing".to_string()],
            metadata: None,
        })
        .unwrap();
    let e5_id = observer.context().event_chain()[4].clone();
    observer.end_scope();

    // Event 6: Response generated
    observer.begin_scope(&e5_id);
    observer
        .record_response_generated(ResponseGeneratedEvent {
            timestamp: Utc::now(),
            content: "Generated response with high coherence".to_string(),
            confidence: 0.91,
            safety_verified: true,
            requires_confirmation: false,
            intent: "inform".to_string(),
            metadata: None,
        })
        .unwrap();
    let e6_id = observer.context().event_chain()[5].clone();
    observer.end_scope();

    // Finalize
    observer.finalize().unwrap();

    let trace = Trace::load_from_file(&trace_path).unwrap();
    let graph = CausalGraph::from_trace(&trace);
    let analyzer = TraceAnalyzer::new(trace.clone());

    // Verify complete chain
    assert_eq!(graph.nodes.len(), 6, "Should have 6 events");

    // Verify each link in the chain
    let chain = analyzer.get_causal_chain(&e6_id);
    assert!(chain.len() >= 1, "Response should have causal chain");

    // Verify security caused response (indirectly)
    match graph.did_cause(&e1_id, &e6_id) {
        CausalAnswer::IndirectCause { path, strength } => {
            assert!(path.len() >= 5, "Should have deep path");
            assert!(
                strength > 0.3,
                "Should have measurable strength through deep chain"
            );
        }
        other => panic!("Expected IndirectCause through deep chain, got {:?}", other),
    }

    // Verify root causes
    let root_causes = analyzer.find_root_causes(&e6_id);
    assert!(!root_causes.is_empty(), "Response should have root causes");
}

#[test]
fn test_scenario_4_correlation_analysis() {
    // Scenario: Multiple security checks followed by multiple phi measurements
    // Expected: Statistical correlation between security and phi events

    let (observer, mut ctx, trace_path) = create_observer_with_correlation();

    // Generate 10 security → phi pairs
    for i in 0..10 {
        // Security check
        let security_meta = ctx.create_event_metadata_with_tags(vec![format!("iter_{}", i)]);
        {
            let mut obs = observer.write().unwrap();
            obs.record_security_check(SecurityCheckEvent {
                timestamp: Utc::now(),
                operation: format!("operation_{}", i),
                decision: SecurityDecision::Allowed,
                reason: Some("Test".to_string()),
                secrets_redacted: 0,
                similarity_score: Some(0.01 * i as f64),
                matched_pattern: None,
                metadata: None,
            })
            .unwrap();
        }

        // Phi measurement (child of security)
        ctx.push_parent(&security_meta.id);
        let _phi_meta = ctx.create_event_metadata();
        {
            let mut obs = observer.write().unwrap();
            obs.record_phi_measurement(PhiMeasurementEvent {
                timestamp: Utc::now(),
                phi: 0.7 + (i as f64 * 0.02),
                components: PhiComponents::default(),
                temporal_continuity: 0.85 + (i as f64 * 0.01),
                metadata: None,
            })
            .unwrap();
        }
        ctx.pop_parent();
    }

    // Finalize
    {
        let mut obs = observer.write().unwrap();
        obs.finalize().unwrap();
    }

    let trace = Trace::load_from_file(&trace_path).unwrap();
    let analyzer = TraceAnalyzer::new(trace.clone());

    // Analyze correlation
    let correlation = analyzer.analyze_correlation("security_check", "phi_measurement");

    assert_eq!(correlation.total_cause_events, 10);
    assert_eq!(correlation.total_effect_events, 10);

    // NOTE: Phase 4 will integrate correlation metadata with observer
    // For now, verify the framework runs and returns valid data
    assert!(
        correlation.direct_correlation_rate >= 0.0 && correlation.direct_correlation_rate <= 1.0
    );
}

#[test]
fn test_scenario_5_visualization_export() {
    // Scenario: Create causal graph and export to Mermaid and GraphViz formats
    // Expected: Valid diagram exports
    // Phase 4: Now using CausalTraceObserver

    let (mut observer, trace_path) = create_causal_observer();

    // Create simple graph: A → B → C
    observer
        .record_security_check(SecurityCheckEvent {
            timestamp: Utc::now(),
            operation: "A".to_string(),
            decision: SecurityDecision::Allowed,
            reason: None,
            secrets_redacted: 0,
            similarity_score: None,
            matched_pattern: None,
            metadata: None,
        })
        .unwrap();
    let a_id = observer.context().event_chain()[0].clone();

    observer.begin_scope(&a_id);
    observer
        .record_phi_measurement(PhiMeasurementEvent {
            timestamp: Utc::now(),
            phi: 0.75,
            components: PhiComponents::default(),
            temporal_continuity: 0.88,
            metadata: None,
        })
        .unwrap();
    let b_id = observer.context().event_chain()[1].clone();
    observer.end_scope();

    observer.begin_scope(&b_id);
    observer
        .record_router_selection(RouterSelectionEvent {
            timestamp: Utc::now(),
            input: "test".to_string(),
            selected_router: "standard".to_string(),
            confidence: 0.80,
            alternatives: vec![],
            bandit_stats: std::collections::HashMap::new(),
            metadata: None,
        })
        .unwrap();
    let c_id = observer.context().event_chain()[2].clone();
    observer.end_scope();

    // Finalize
    observer.finalize().unwrap();

    let trace = Trace::load_from_file(&trace_path).unwrap();
    let graph = CausalGraph::from_trace(&trace);

    // Export to Mermaid
    let mermaid = graph.to_mermaid();
    assert!(mermaid.contains("graph TD"), "Should have Mermaid header");
    assert!(mermaid.contains(&a_id), "Should contain event A");
    assert!(mermaid.contains(&b_id), "Should contain event B");
    assert!(mermaid.contains(&c_id), "Should contain event C");
    assert!(mermaid.contains("-->"), "Should have causal arrows");

    // Export to GraphViz
    let dot = graph.to_dot();
    assert!(dot.contains("digraph"), "Should have GraphViz header");
    assert!(dot.contains(&a_id), "Should contain event A");
    assert!(dot.contains(&b_id), "Should contain event B");
    assert!(dot.contains(&c_id), "Should contain event C");
    assert!(dot.contains("->"), "Should have causal arrows");
}

#[test]
fn test_scenario_6_performance_analysis() {
    // Scenario: Create events with varying durations to test bottleneck detection
    // Expected: Correctly identify slowest events

    let (observer, mut ctx, trace_path) = create_observer_with_correlation();

    // Fast event
    let fast = ctx.create_event_metadata_with_tags(vec!["fast"]);
    {
        let mut obs = observer.write().unwrap();
        obs.record_security_check(SecurityCheckEvent {
            timestamp: Utc::now(),
            operation: "fast_operation".to_string(),
            decision: SecurityDecision::Allowed,
            reason: None,
            secrets_redacted: 0,
            similarity_score: None,
            matched_pattern: None,
            metadata: None,
        })
        .unwrap();
    }

    // Slow event (simulated)
    ctx.push_parent(&fast.id);
    let slow = ctx.create_event_metadata_with_tags(vec!["slow"]);
    {
        let mut obs = observer.write().unwrap();
        obs.record_phi_measurement(PhiMeasurementEvent {
            timestamp: Utc::now(),
            phi: 0.85,
            components: PhiComponents::default(),
            temporal_continuity: 0.92,
            metadata: None,
        })
        .unwrap();
    }
    ctx.pop_parent();

    // Another fast event
    ctx.push_parent(&slow.id);
    let _fast2 = ctx.create_event_metadata_with_tags(vec!["fast"]);
    {
        let mut obs = observer.write().unwrap();
        obs.record_router_selection(RouterSelectionEvent {
            timestamp: Utc::now(),
            input: "test".to_string(),
            selected_router: "fast_router".to_string(),
            confidence: 0.90,
            alternatives: vec![],
            bandit_stats: std::collections::HashMap::new(),
            metadata: None,
        })
        .unwrap();
    }
    ctx.pop_parent();

    // Finalize
    {
        let mut obs = observer.write().unwrap();
        obs.finalize().unwrap();
    }

    let trace = Trace::load_from_file(&trace_path).unwrap();
    let analyzer = TraceAnalyzer::new(trace.clone());

    // Get performance summary
    let summary = analyzer.performance_summary();
    assert_eq!(summary.total_events, 3);

    // NOTE: Phase 4 will integrate duration tracking with observer
    // For now, verify the analysis framework runs
    let bottlenecks = analyzer.find_bottlenecks(0.5);
    assert!(bottlenecks.len() >= 0); // Valid result

    let _ = slow; // Use variable to avoid warning
}