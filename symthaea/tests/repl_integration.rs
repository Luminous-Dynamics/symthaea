//! Integration tests for the repl/ module.
//!
//! Tests cover:
//! - ReplSession creation, warmup, processing, and reset
//! - Consciousness state inspection
//! - History and stats tracking
//! - ReplOrchestrator creation and command handling
//! - ConsciousnessTracer lifecycle
//! - MetricsAggregator initial state

use symthaea::repl::observability_hooks::MetricsAggregator;
use symthaea::repl::{
    ConsciousnessTracer, OrchestratorConfig, ReplOrchestrator, ReplSession, ReplSessionConfig,
};

// ============================================================================
// ReplSession Creation Tests
// ============================================================================

#[test]
fn test_session_creation_default_config() {
    let session = ReplSession::new(ReplSessionConfig::default());
    assert!(session.is_ok(), "Default config session should succeed");
    let session = session.unwrap();
    assert_eq!(session.stats().total_interactions, 0);
    assert_eq!(session.stats().total_cycles, 0);
}

#[test]
fn test_session_warmup() {
    let mut session = ReplSession::new(ReplSessionConfig::default()).unwrap();
    let prediction_error = session.warmup(5);
    assert!(
        prediction_error.is_finite(),
        "Warmup prediction error should be finite, got {}",
        prediction_error
    );
}

// ============================================================================
// ReplSession Processing Tests
// ============================================================================

#[test]
fn test_session_process_increments_stats() {
    let mut session = ReplSession::new(ReplSessionConfig::default()).unwrap();
    session.warmup(3);

    let result1 = session.process("Hello");
    assert!(result1.is_ok(), "First process should succeed");

    let result2 = session.process("How are you?");
    assert!(result2.is_ok(), "Second process should succeed");

    assert_eq!(
        session.stats().total_interactions,
        2,
        "Should have 2 total interactions"
    );
}

#[test]
fn test_session_process_returns_response() {
    let mut session = ReplSession::new(ReplSessionConfig::default()).unwrap();
    session.warmup(3);

    let result = session.process("What is consciousness?").unwrap();
    assert!(
        !result.response.is_empty(),
        "Process should return a non-empty response"
    );
}

#[test]
fn test_session_consciousness_state() {
    let mut session = ReplSession::new(ReplSessionConfig::default()).unwrap();
    session.warmup(3);
    let _ = session.process("test input");

    let state = session.consciousness_state();
    assert!(
        state.unified_psi.is_finite(),
        "Psi should be finite after processing"
    );
    assert!(
        state.temporal_coherence.is_finite(),
        "Temporal coherence should be finite after processing"
    );
}

// ============================================================================
// History & Stats Tests
// ============================================================================

#[test]
fn test_session_history_records_turns() {
    let mut session = ReplSession::new(ReplSessionConfig::default()).unwrap();
    session.warmup(3);

    for msg in &["Alpha", "Beta", "Gamma"] {
        let _ = session.process(msg);
    }

    assert_eq!(session.history().len(), 3, "History should contain 3 turns");
}

#[test]
fn test_session_reset_clears_state() {
    let mut session = ReplSession::new(ReplSessionConfig::default()).unwrap();
    session.warmup(3);
    let _ = session.process("something");
    assert!(!session.history().is_empty());

    session.reset();
    assert!(
        session.history().is_empty(),
        "History should be empty after reset"
    );
}

#[test]
fn test_session_stats_track_cycles() {
    let config = ReplSessionConfig {
        cycles_per_input: 5,
        ..Default::default()
    };
    let mut session = ReplSession::new(config).unwrap();
    session.warmup(3);

    let _ = session.process("test");
    assert!(
        session.stats().total_cycles >= 5,
        "Should have at least cycles_per_input cycles, got {}",
        session.stats().total_cycles
    );
}

// ============================================================================
// ReplOrchestrator Tests
// ============================================================================

#[test]
fn test_orchestrator_creation() {
    let orch = ReplOrchestrator::new(OrchestratorConfig::default());
    assert!(orch.is_ok(), "Default orchestrator creation should succeed");
}

#[test]
fn test_orchestrator_command_handling() {
    let mut orch = ReplOrchestrator::new(OrchestratorConfig::default()).unwrap();

    let help_result = orch.handle_command("/help");
    assert!(help_result.is_some(), "/help should return Some");

    let quit_result = orch.handle_command("/quit");
    assert!(quit_result.is_some(), "/quit should return Some");
    assert!(orch.should_shutdown(), "/quit should set shutdown flag");

    // Non-command input returns None
    let mut orch2 = ReplOrchestrator::new(OrchestratorConfig::default()).unwrap();
    let plain = orch2.handle_command("hello");
    assert!(plain.is_none(), "Plain text should return None");
}

// ============================================================================
// Observability Tests
// ============================================================================

#[test]
fn test_consciousness_tracer_lifecycle() {
    let tracer = ConsciousnessTracer::new("test-session-001");
    assert_eq!(tracer.session_id(), "test-session-001");
    assert!(
        tracer.traces().is_empty(),
        "Tracer should start with no traces"
    );
}

#[test]
fn test_metrics_aggregator_empty() {
    let aggregator = MetricsAggregator::new("test-session-002");
    assert!(
        aggregator.phi_history().is_empty(),
        "Phi history should start empty"
    );
    assert!(
        aggregator.coherence_history().is_empty(),
        "Coherence history should start empty"
    );
}
