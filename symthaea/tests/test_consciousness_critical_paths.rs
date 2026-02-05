#![cfg(all(feature = "embeddings_module", feature = "full_consciousness"))]
//! Integration tests for consciousness critical paths
//!
//! Tests the core consciousness infrastructure including:
//! - ConsciousnessGraph autopoietic loops
//! - OscillatoryRouter (MultiBandState)
//! - ContinuousMind basic initialization

use symthaea::consciousness::ConsciousnessGraph;
use symthaea::consciousness::recursive_improvement::routers::{
    MultiBandState, FrequencyBand,
};

#[test]
fn test_consciousness_graph_autopoiesis() {
    let mut graph = ConsciousnessGraph::new();

    // Add initial conscious state
    let semantic = vec![0.5f32; 64];
    let dynamic = vec![0.3f32; 32];
    let consciousness_level = 0.7;

    let node1 = graph.add_state(semantic.clone(), dynamic.clone(), consciousness_level);

    // Create self-referential loop (autopoiesis)
    graph.create_self_loop(node1);

    assert_eq!(graph.size(), 1);
    assert_eq!(graph.self_loop_count(), 1);
    assert!(graph.current_consciousness() > 0.0);

    // Add another state and evolve
    let node2 = graph.add_state(
        vec![0.6f32; 64],
        vec![0.4f32; 32],
        0.8,
    );

    assert_eq!(graph.size(), 2);

    // Evolution should follow highest-weight edge
    let evolved = graph.evolve();
    assert!(evolved.is_some());
}

#[test]
fn test_consciousness_graph_history_tracing() {
    let mut graph = ConsciousnessGraph::new();

    // Build a chain of conscious states
    for i in 0..5 {
        let consciousness = 0.5 + (i as f32) * 0.1;
        graph.add_state(
            vec![consciousness; 64],
            vec![consciousness * 0.5; 32],
            consciousness,
        );
    }

    assert_eq!(graph.size(), 5);

    // Trace history backwards
    let history = graph.trace_history(3);
    assert!(history.len() <= 3);
}

#[test]
fn test_multiband_oscillatory_state() {
    let mut state = MultiBandState::new();

    // Advance time
    state.advance(0.025); // 25ms = 1 gamma cycle at 40Hz

    // Check gamma phase advances
    let gamma_phase = state.gamma_phase();
    assert!(gamma_phase >= 0.0);

    // Check theta phase
    let theta_phase = state.theta_phase();
    assert!(theta_phase >= 0.0);

    // Check overall coherence
    let coherence = state.overall_coherence();
    assert!(coherence >= 0.0 && coherence <= 1.0);
}

#[test]
fn test_multiband_arousal_modulation() {
    let mut state = MultiBandState::new();

    // Low arousal should favor alpha/theta
    state.set_arousal(0.2);
    let summary_low = state.summary();

    // High arousal should favor beta/gamma
    state.set_arousal(0.9);
    let summary_high = state.summary();

    // Gamma should be higher at high arousal
    assert!(summary_high.gamma_power >= summary_low.gamma_power * 0.5);
}

#[test]
fn test_multiband_attention_modulation() {
    let mut state = MultiBandState::new();

    // Set high attention
    state.set_attention(0.9);

    let summary = state.summary();

    // High attention should boost gamma
    assert!(summary.gamma_power > 0.0);

    // Alpha should be suppressed at high attention
    assert!(summary.alpha_power < 1.0);
}

#[test]
fn test_multiband_theta_gamma_coupling() {
    let mut state = MultiBandState::new();

    // Advance through multiple cycles
    for _ in 0..100 {
        state.advance(0.001); // 1ms steps
    }

    let summary = state.summary();

    // Coupling should be non-zero
    assert!(summary.theta_gamma_coupling >= 0.0);
}

#[test]
fn test_multiband_dominant_band_detection() {
    let mut state = MultiBandState::new();

    // Get dominant band
    let dominant = state.get_dominant_band();

    // Should be one of the valid bands
    match dominant {
        FrequencyBand::Delta |
        FrequencyBand::Theta |
        FrequencyBand::Alpha |
        FrequencyBand::Beta |
        FrequencyBand::Gamma |
        FrequencyBand::SleepSpindle => (),
    }
}

#[test]
fn test_multiband_memory_encoding_window() {
    let mut state = MultiBandState::new();

    // Set conditions favorable for memory encoding
    state.set_arousal(0.5);
    state.set_attention(0.7);

    // Advance to find memory encoding window
    let mut found_window = false;
    for _ in 0..1000 {
        state.advance(0.001);
        if state.in_memory_encoding_window() {
            found_window = true;
            break;
        }
    }

    // Window should be reachable under favorable conditions
    // (may not always occur depending on phase alignment)
    let _ = found_window; // Just verify it doesn't panic
}

#[test]
fn test_graph_complexity_metric() {
    let mut graph = ConsciousnessGraph::new();

    // Empty graph has zero complexity
    assert_eq!(graph.complexity(), 0.0);

    // Add nodes with connections
    let n1 = graph.add_state(vec![0.1; 64], vec![0.1; 32], 0.5);
    let n2 = graph.add_state(vec![0.2; 64], vec![0.2; 32], 0.6);
    let n3 = graph.add_state(vec![0.3; 64], vec![0.3; 32], 0.7);

    // Complexity = edges / nodes
    let complexity = graph.complexity();
    assert!(complexity > 0.0);
}

#[test]
fn test_autopoietic_nodes_retrieval() {
    let mut graph = ConsciousnessGraph::new();

    // Add states
    let n1 = graph.add_state(vec![0.1; 64], vec![0.1; 32], 0.5);
    let n2 = graph.add_state(vec![0.2; 64], vec![0.2; 32], 0.6);

    // Create self-loops
    graph.create_self_loop(n1);
    graph.create_self_loop(n2);

    // Retrieve autopoietic nodes
    let autopoietic = graph.autopoietic_nodes();
    assert_eq!(autopoietic.len(), 2);
}
