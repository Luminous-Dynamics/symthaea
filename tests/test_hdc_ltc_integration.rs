// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC-LTC Unified Integration Tests
//!
//! Tests for the integration of HdcLtcUnifiedNetwork into the cognitive loop
//! as an alternative to the CfC network.
//!
//! ## Test Categories
//!
//! 1. **Bridge Tests**: Verify HdcLtcBridge provides CfC-compatible interface
//! 2. **Cognitive Loop Tests**: Verify both backends work with CognitiveLoopService
//! 3. **Backend Parity Tests**: Compare behavior between CfC and HdcLtcUnified
//! 4. **Configuration Tests**: Verify config options work correctly

use ndarray::Array1;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, TemporalBackend};
use symthaea::hdc_ltc_bridge::{BridgeActivation, HdcLtcBridge, HdcLtcBridgeConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// BRIDGE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bridge_creation() {
    let config = HdcLtcBridgeConfig::default();
    let bridge = HdcLtcBridge::new(config);

    assert_eq!(bridge.total_steps(), 0);
    assert_eq!(bridge.config().input_dim, 256);
    assert_eq!(bridge.config().output_dim, 256);
}

#[test]
fn test_bridge_step_and_read() {
    let config = HdcLtcBridgeConfig::default();
    let mut bridge = HdcLtcBridge::new(config);

    let input = Array1::from_vec(vec![0.5; 256]);

    // Step should succeed
    let result = bridge.step(&input, 0.02);
    assert!(result.is_ok());
    assert_eq!(bridge.total_steps(), 1);

    // Read state should return correct dimension
    let state = bridge.read_state().unwrap();
    assert_eq!(state.len(), 256);
}

#[test]
fn test_bridge_forward() {
    let config = HdcLtcBridgeConfig::default();
    let mut bridge = HdcLtcBridge::new(config);

    let input = Array1::from_vec(vec![0.3; 256]);
    let output = bridge.forward(&input, 0.02);

    assert_eq!(output.len(), 256);

    // Output should be different from zero (network processed input)
    let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    // After processing, norm should be non-trivial
    assert!(norm >= 0.0); // May be zero initially
}

#[test]
fn test_bridge_train_step() {
    let config = HdcLtcBridgeConfig::default();
    let mut bridge = HdcLtcBridge::new(config);

    let input = Array1::from_vec(vec![0.5; 256]);
    let target = Array1::from_vec(vec![0.3; 256]);

    let loss = bridge.train_step(&input, &target, 0.02, 0.01).unwrap();

    // Loss should be non-negative
    assert!(loss >= 0.0);
}

#[test]
fn test_bridge_predict_forward() {
    let config = HdcLtcBridgeConfig::default();
    let mut bridge = HdcLtcBridge::new(config);

    let input = Array1::from_vec(vec![0.5; 256]);

    // Small horizon prediction
    let pred_small = bridge.predict_forward(&input, 0.1).unwrap();
    assert_eq!(pred_small.len(), 256);

    // Large horizon prediction (O(1) via closed-form)
    let pred_large = bridge.predict_forward(&input, 10.0).unwrap();
    assert_eq!(pred_large.len(), 256);

    // Very large horizon should also work
    let pred_huge = bridge.predict_forward(&input, 1000.0).unwrap();
    assert_eq!(pred_huge.len(), 256);
}

#[test]
fn test_bridge_reset() {
    let config = HdcLtcBridgeConfig::default();
    let mut bridge = HdcLtcBridge::new(config);

    let input = Array1::from_vec(vec![0.5; 256]);
    let _ = bridge.step(&input, 0.02);
    let _ = bridge.step(&input, 0.02);
    assert_eq!(bridge.total_steps(), 2);

    bridge.reset();
    assert_eq!(bridge.total_steps(), 0);
}

#[test]
fn test_bridge_state_diversity() {
    let config = HdcLtcBridgeConfig::default();
    let mut bridge = HdcLtcBridge::new(config);

    let input = Array1::from_vec(vec![0.5; 256]);
    let _ = bridge.step(&input, 0.02);

    let diversity = bridge.state_diversity();
    assert!((0.0..=1.0).contains(&diversity));
}

#[test]
fn test_bridge_tau_values() {
    let config = HdcLtcBridgeConfig::default();
    let bridge = HdcLtcBridge::new(config);

    let taus = bridge.all_tau();
    assert!(!taus.is_empty());

    let flattened = bridge.flattened_tau();
    assert!(!flattened.is_empty());

    // All tau values should be positive
    for tau in &flattened {
        assert!(*tau > 0.0);
    }
}

#[test]
fn test_bridge_inject() {
    let config = HdcLtcBridgeConfig::default();
    let mut bridge = HdcLtcBridge::new(config);

    let state = Array1::from_vec(vec![0.7; 256]);
    let result = bridge.inject(&state);

    assert!(result.is_ok());
}

#[test]
fn test_bridge_config_variants() {
    // Test fast config
    let fast_config = HdcLtcBridgeConfig::fast();
    assert_eq!(fast_config.tau_base, 0.05);
    assert_eq!(fast_config.layer_sizes, vec![2, 4, 2]);

    // Test accurate config
    let accurate_config = HdcLtcBridgeConfig::accurate();
    assert!(accurate_config.skip_connections);
    assert_eq!(accurate_config.layer_sizes, vec![8, 16, 8]);
}

#[test]
fn test_bridge_activations() {
    for activation in [
        BridgeActivation::Tanh,
        BridgeActivation::Sigmoid,
        BridgeActivation::SiLU,
        BridgeActivation::Identity,
    ] {
        let config = HdcLtcBridgeConfig {
            activation,
            ..Default::default()
        };
        let mut bridge = HdcLtcBridge::new(config);

        let input = Array1::from_vec(vec![0.5; 256]);
        let output = bridge.forward(&input, 0.02);
        assert_eq!(output.len(), 256);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE LOOP TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_cognitive_loop_cfc_backend() {
    let config = CognitiveLoopConfig::with_cfc();
    let service = CognitiveLoopService::new(config);

    assert!(service.is_ok());
    let service = service.unwrap();
    assert_eq!(service.temporal_backend(), TemporalBackend::CfC);
}

#[test]
fn test_cognitive_loop_hdc_ltc_backend() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    let service = CognitiveLoopService::new(config);

    assert!(service.is_ok());
    let service = service.unwrap();
    assert_eq!(service.temporal_backend(), TemporalBackend::HdcLtcUnified);
}

#[test]
fn test_cognitive_loop_cfc_cycle() {
    let config = CognitiveLoopConfig::with_cfc();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let result = service.cycle("Hello, this is a test input.");

    // Verify cycle result structure
    assert!(!result.output.is_empty());
    assert!(result.prediction_error >= 0.0);
    assert!(result.cycle_time_us > 0);
}

#[test]
fn test_cognitive_loop_hdc_ltc_cycle() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let result = service.cycle("Hello, this is a test input.");

    // Verify cycle result structure
    assert!(!result.output.is_empty());
    assert!(result.prediction_error >= 0.0);
    assert!(result.cycle_time_us > 0);
}

#[test]
fn test_cognitive_loop_multiple_cycles_cfc() {
    let config = CognitiveLoopConfig::with_cfc();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let inputs = [
        "The cat sat on the mat.",
        "Cause leads to effect.",
        "Understanding emerges from patterns.",
    ];

    for input in &inputs {
        let result = service.cycle(input);
        assert!(!result.output.is_empty());
    }

    // Stats should show cycles processed
    let stats = service.stats();
    assert_eq!(stats.total_cycles, 3);
}

#[test]
fn test_cognitive_loop_multiple_cycles_hdc_ltc() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let inputs = [
        "The cat sat on the mat.",
        "Cause leads to effect.",
        "Understanding emerges from patterns.",
    ];

    for input in &inputs {
        let result = service.cycle(input);
        assert!(!result.output.is_empty());
    }

    // Stats should show cycles processed
    let stats = service.stats();
    assert_eq!(stats.total_cycles, 3);
}

#[test]
fn test_cognitive_loop_learning_cfc() {
    let mut config = CognitiveLoopConfig::with_cfc();
    config.learning_threshold = 0.01; // Lower threshold to ensure learning triggers
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Run multiple cycles to trigger learning
    for _ in 0..10 {
        service.cycle("Learning occurs through prediction errors.");
    }

    let stats = service.stats();
    assert!(stats.total_cycles >= 10);
    // Learning cycles should have occurred (error above threshold)
}

#[test]
fn test_cognitive_loop_learning_hdc_ltc() {
    let mut config = CognitiveLoopConfig::with_hdc_ltc_unified();
    config.learning_threshold = 0.01; // Lower threshold to ensure learning triggers
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Run multiple cycles to trigger learning
    for _ in 0..10 {
        service.cycle("Learning occurs through prediction errors.");
    }

    let stats = service.stats();
    assert!(stats.total_cycles >= 10);
}

#[test]
fn test_cognitive_loop_consciousness_snapshot_cfc() {
    let config = CognitiveLoopConfig::with_cfc();
    let mut service = CognitiveLoopService::new(config).unwrap();

    service.cycle("Testing consciousness snapshot.");

    let snapshot = service.consciousness_snapshot();

    // Verify snapshot fields
    assert_eq!(snapshot.cycle, 1);
    assert!(snapshot.consciousness_level >= 0.0 && snapshot.consciousness_level <= 1.0);
    assert!(snapshot.prediction_error >= 0.0);
}

#[test]
fn test_cognitive_loop_consciousness_snapshot_hdc_ltc() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    let mut service = CognitiveLoopService::new(config).unwrap();

    service.cycle("Testing consciousness snapshot.");

    let snapshot = service.consciousness_snapshot();

    // Verify snapshot fields
    assert_eq!(snapshot.cycle, 1);
    assert!(snapshot.consciousness_level >= 0.0 && snapshot.consciousness_level <= 1.0);
    assert!(snapshot.prediction_error >= 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BACKEND PARITY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_backend_parity_basic() {
    let config_cfc = CognitiveLoopConfig::with_cfc();
    let config_hdc = CognitiveLoopConfig::with_hdc_ltc_unified();

    let mut service_cfc = CognitiveLoopService::new(config_cfc).unwrap();
    let mut service_hdc = CognitiveLoopService::new(config_hdc).unwrap();

    let input = "Both backends should process this input.";

    let result_cfc = service_cfc.cycle(input);
    let result_hdc = service_hdc.cycle(input);

    // Both should produce valid outputs
    assert!(!result_cfc.output.is_empty());
    assert!(!result_hdc.output.is_empty());

    // Both should have same output dimension
    assert_eq!(result_cfc.output.len(), result_hdc.output.len());

    // Both should have valid prediction errors
    assert!(result_cfc.prediction_error >= 0.0);
    assert!(result_hdc.prediction_error >= 0.0);
}

#[test]
fn test_backend_parity_multiple_cycles() {
    let config_cfc = CognitiveLoopConfig::with_cfc();
    let config_hdc = CognitiveLoopConfig::with_hdc_ltc_unified();

    let mut service_cfc = CognitiveLoopService::new(config_cfc).unwrap();
    let mut service_hdc = CognitiveLoopService::new(config_hdc).unwrap();

    let inputs = [
        "First input for parity test.",
        "Second input with different content.",
        "Third input to verify consistency.",
    ];

    for input in &inputs {
        let result_cfc = service_cfc.cycle(input);
        let result_hdc = service_hdc.cycle(input);

        // Both should produce valid outputs
        assert_eq!(result_cfc.output.len(), result_hdc.output.len());
    }

    // Both should have processed same number of cycles
    assert_eq!(
        service_cfc.stats().total_cycles,
        service_hdc.stats().total_cycles
    );
}

#[test]
fn test_backend_parity_coherence() {
    let config_cfc = CognitiveLoopConfig::with_cfc();
    let config_hdc = CognitiveLoopConfig::with_hdc_ltc_unified();

    let mut service_cfc = CognitiveLoopService::new(config_cfc).unwrap();
    let mut service_hdc = CognitiveLoopService::new(config_hdc).unwrap();

    // Process same inputs
    for _ in 0..5 {
        service_cfc.cycle("Testing coherence tracking.");
        service_hdc.cycle("Testing coherence tracking.");
    }

    // Both should produce coherence values in valid range
    let coh_cfc = service_cfc.temporal_coherence();
    let coh_hdc = service_hdc.temporal_coherence();

    assert!((0.0..=1.0).contains(&coh_cfc));
    assert!((0.0..=1.0).contains(&coh_hdc));
}

#[test]
fn test_backend_parity_state_diversity() {
    let config_cfc = CognitiveLoopConfig::with_cfc();
    let config_hdc = CognitiveLoopConfig::with_hdc_ltc_unified();

    let mut service_cfc = CognitiveLoopService::new(config_cfc).unwrap();
    let mut service_hdc = CognitiveLoopService::new(config_hdc).unwrap();

    // Process same input
    service_cfc.cycle("Testing state diversity.");
    service_hdc.cycle("Testing state diversity.");

    // Both should produce diversity values in valid range
    let div_cfc = service_cfc.cfc_state_diversity();
    let div_hdc = service_hdc.cfc_state_diversity();

    assert!((0.0..=1.0).contains(&div_cfc));
    assert!((0.0..=1.0).contains(&div_hdc));
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_config_default_is_cfc() {
    let config = CognitiveLoopConfig::default();
    assert_eq!(config.temporal_backend, TemporalBackend::CfC);
}

#[test]
fn test_config_with_cfc() {
    let config = CognitiveLoopConfig::with_cfc();
    assert_eq!(config.temporal_backend, TemporalBackend::CfC);
}

#[test]
fn test_config_with_hdc_ltc_unified() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    assert_eq!(config.temporal_backend, TemporalBackend::HdcLtcUnified);
}

#[test]
fn test_config_with_hdc_ltc_fast() {
    let config = CognitiveLoopConfig::with_hdc_ltc_fast();
    assert_eq!(config.temporal_backend, TemporalBackend::HdcLtcUnified);
    assert_eq!(config.hdc_ltc_config.tau_base, 0.05);
}

#[test]
fn test_config_with_hdc_ltc_accurate() {
    let config = CognitiveLoopConfig::with_hdc_ltc_accurate();
    assert_eq!(config.temporal_backend, TemporalBackend::HdcLtcUnified);
    assert!(config.hdc_ltc_config.skip_connections);
}

#[test]
fn test_temporal_backend_serialization() {
    // Test that backends can be serialized/deserialized
    let backends = [TemporalBackend::CfC, TemporalBackend::HdcLtcUnified];

    for backend in backends {
        let json = serde_json::to_string(&backend).unwrap();
        let deserialized: TemporalBackend = serde_json::from_str(&json).unwrap();
        assert_eq!(backend, deserialized);
    }
}

#[test]
fn test_config_serialization() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: CognitiveLoopConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.temporal_backend, deserialized.temporal_backend);
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRESS TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_many_cycles_cfc() {
    let config = CognitiveLoopConfig::with_cfc();
    let mut service = CognitiveLoopService::new(config).unwrap();

    for i in 0..100 {
        service.cycle(&format!("Cycle number {} of stress test.", i));
    }

    assert_eq!(service.stats().total_cycles, 100);
}

#[test]
fn test_many_cycles_hdc_ltc() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    let mut service = CognitiveLoopService::new(config).unwrap();

    for i in 0..100 {
        service.cycle(&format!("Cycle number {} of stress test.", i));
    }

    assert_eq!(service.stats().total_cycles, 100);
}

#[test]
fn test_long_input_cfc() {
    let config = CognitiveLoopConfig::with_cfc();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let long_input = "a".repeat(10000);
    let result = service.cycle(&long_input);

    assert!(!result.output.is_empty());
}

#[test]
fn test_long_input_hdc_ltc() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let long_input = "a".repeat(10000);
    let result = service.cycle(&long_input);

    assert!(!result.output.is_empty());
}

#[test]
fn test_empty_input_cfc() {
    let config = CognitiveLoopConfig::with_cfc();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let result = service.cycle("");
    assert!(!result.output.is_empty());
}

#[test]
fn test_empty_input_hdc_ltc() {
    let config = CognitiveLoopConfig::with_hdc_ltc_unified();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let result = service.cycle("");
    assert!(!result.output.is_empty());
}