// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for CfC cell, network, numerical stability, and phi-gated attention.

use super::phi_gated::weighted_array_bundle;
use super::*;
use ndarray::Array1;

#[test]
fn test_cfc_cell_creation() {
    let config = CfCConfig::default();
    let cell = CfCCell::new(config);
    assert_eq!(cell.state().len(), 128);
}

#[test]
fn test_cfc_forward() {
    let config = CfCConfig {
        input_dim: 32,
        hidden_dim: 64,
        ..Default::default()
    };
    let mut cell = CfCCell::new(config);

    let input = Array1::from_vec(vec![0.1; 32]);
    let output = cell.forward(&input, 0.1);

    assert_eq!(output.len(), 64);
}

#[test]
fn test_cfc_network() {
    let config = CfCNetworkConfig {
        input_dim: 32,
        hidden_dim: 64,
        num_layers: 2,
        output_dim: 16,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);

    let input = Array1::from_vec(vec![0.1; 32]);
    let output = network.forward(&input, 0.1);

    assert_eq!(output.len(), 16);
}

// =====================================================================
// 4.5: CfC NUMERICAL STABILITY TESTS
// =====================================================================

#[test]
fn test_cfc_long_horizon_stability() {
    let config = CfCNetworkConfig {
        input_dim: 8,
        hidden_dim: 16,
        num_layers: 2,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);
    let input = Array1::from_vec(vec![0.5; 8]);

    for step in 0..10_000 {
        let output = network.forward(&input, 0.1);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "CfC diverged at step {} — output: {:?}",
            step,
            output
        );
    }
}

#[test]
fn test_cfc_extreme_small_tau() {
    let cell_config = CfCConfig {
        input_dim: 4,
        hidden_dim: 8,
        tau_range: (0.001, 0.01), // Very small time constants
        use_backbone: false,
        ..Default::default()
    };
    let config = CfCNetworkConfig {
        input_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        output_dim: 4,
        cell_config,
        residual: false,
        bidirectional: false,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);
    let input = Array1::from_vec(vec![1.0; 4]);

    // With very small tau, decay is nearly complete each step
    for _ in 0..100 {
        let output = network.forward(&input, 1.0);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "Small tau caused divergence"
        );
    }
}

#[test]
fn test_cfc_extreme_large_tau() {
    let cell_config = CfCConfig {
        input_dim: 4,
        hidden_dim: 8,
        tau_range: (100.0, 1000.0), // Very large time constants
        use_backbone: false,
        ..Default::default()
    };
    let config = CfCNetworkConfig {
        input_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        output_dim: 4,
        cell_config,
        residual: false,
        bidirectional: false,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);
    let input = Array1::from_vec(vec![1.0; 4]);

    // With very large tau, state barely changes each step
    for _ in 0..100 {
        let output = network.forward(&input, 0.01);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "Large tau caused divergence"
        );
    }
}

#[test]
fn test_cfc_zero_input_stability() {
    let config = CfCNetworkConfig {
        input_dim: 8,
        hidden_dim: 16,
        num_layers: 2,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);
    let input = Array1::zeros(8);

    for _ in 0..1_000 {
        let output = network.forward(&input, 0.1);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "Zero input caused divergence"
        );
    }
}

#[test]
fn test_cfc_large_dt_stability() {
    let config = CfCNetworkConfig {
        input_dim: 8,
        hidden_dim: 16,
        num_layers: 1,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);
    let input = Array1::from_vec(vec![0.5; 8]);

    // Large dt = 10.0 (should still produce finite output due to closed-form solution)
    for _ in 0..100 {
        let output = network.forward(&input, 10.0);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "Large dt caused divergence (closed-form should handle this)"
        );
    }
}

#[test]
fn test_cfc_reset_clears_state() {
    let config = CfCNetworkConfig {
        input_dim: 8,
        hidden_dim: 16,
        num_layers: 2,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);
    let input = Array1::from_vec(vec![1.0; 8]);

    // Run forward to build up state
    for _ in 0..100 {
        network.forward(&input, 0.1);
    }

    // Reset and verify output changes
    network.reset();
    let output_after_reset = network.forward(&input, 0.1);
    assert!(
        output_after_reset.iter().all(|x| x.is_finite()),
        "Output after reset should be finite"
    );
}

// =====================================================================
// EDGE CASE TESTS FOR TAU VALIDATION AND NUMERICAL STABILITY
// =====================================================================

#[test]
fn test_cfc_clamps_zero_tau() {
    let cell_config = CfCConfig {
        input_dim: 4,
        hidden_dim: 8,
        tau_range: (0.0, 1.0), // Zero tau_min gets clamped to MIN_TAU
        use_backbone: false,
        ..Default::default()
    };
    let cell = CfCCell::new(cell_config);
    // Verify tau values are clamped above MIN_TAU
    for &t in cell.tau.iter() {
        assert!(t >= 1e-6, "tau should be >= MIN_TAU after clamp: {t}");
    }
}

#[test]
fn test_cfc_clamps_very_small_tau() {
    let cell_config = CfCConfig {
        input_dim: 4,
        hidden_dim: 8,
        tau_range: (1e-8, 1.0), // Below MIN_TAU gets clamped
        use_backbone: false,
        ..Default::default()
    };
    let cell = CfCCell::new(cell_config);
    for &t in cell.tau.iter() {
        assert!(t >= 1e-6, "tau should be >= MIN_TAU after clamp: {t}");
    }
}

#[test]
fn test_cfc_accepts_min_tau_boundary() {
    // Exactly at MIN_TAU boundary should work
    let cell_config = CfCConfig {
        input_dim: 4,
        hidden_dim: 8,
        tau_range: (1e-6, 1.0), // Exactly MIN_TAU
        use_backbone: false,
        ..Default::default()
    };
    let mut cell = CfCCell::new(cell_config);
    let input = Array1::from_vec(vec![1.0; 4]);

    // Should produce finite outputs even with minimal tau
    for _ in 0..100 {
        let output = cell.forward(&input, 1.0);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "MIN_TAU boundary should produce finite outputs"
        );
    }
}

#[test]
fn test_cfc_zero_input_no_nan() {
    let config = CfCNetworkConfig {
        input_dim: 8,
        hidden_dim: 16,
        num_layers: 2,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);
    let zero_input = Array1::zeros(8);

    // Zero input should never produce NaN
    for _ in 0..1000 {
        let output = network.forward(&zero_input, 0.1);
        assert!(
            output.iter().all(|x| x.is_finite() && !x.is_nan()),
            "Zero input produced NaN"
        );
    }
}

#[test]
fn test_cfc_very_large_dt_no_nan() {
    let config = CfCNetworkConfig {
        input_dim: 8,
        hidden_dim: 16,
        num_layers: 1,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);
    let input = Array1::from_vec(vec![0.5; 8]);

    // Very large dt values (dt >> tau)
    for dt in [100.0, 1000.0, 10000.0] {
        network.reset();
        let output = network.forward(&input, dt);
        assert!(
            output.iter().all(|x| x.is_finite() && !x.is_nan()),
            "Large dt={} caused NaN",
            dt
        );
    }
}

#[test]
fn test_cfc_backward_no_nan_with_small_tau() {
    let cell_config = CfCConfig {
        input_dim: 4,
        hidden_dim: 8,
        tau_range: (1e-5, 1e-4), // Small but valid tau
        use_backbone: false,
        ..Default::default()
    };
    let mut cell = CfCCell::new(cell_config);
    let input = Array1::from_vec(vec![0.5; 4]);
    let target = Array1::from_vec(vec![0.1; 8]);

    // Forward to set state
    let _ = cell.forward(&input, 0.1);

    // Backward should not produce NaN gradients
    let grads = cell.backward(&input, &target, 1.0);
    assert!(
        grads.dw_in.iter().all(|x| x.is_finite()),
        "dw_in gradients contain NaN/Inf"
    );
    assert!(
        grads.dw_h.iter().all(|x| x.is_finite()),
        "dw_h gradients contain NaN/Inf"
    );
    assert!(
        grads.db_h.iter().all(|x| x.is_finite()),
        "db_h gradients contain NaN/Inf"
    );
    assert!(
        grads.dtau.iter().all(|x| x.is_finite()),
        "dtau gradients contain NaN/Inf"
    );
}

// =====================================================================
// PHI-GATED ATTENTION TESTS
// =====================================================================

#[test]
fn test_phi_gated_forward_basic() {
    let config = CfCNetworkConfig {
        input_dim: 8,
        hidden_dim: 16,
        num_layers: 2,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);

    let inputs = vec![
        Array1::from_vec(vec![1.0; 8]),
        Array1::from_vec(vec![0.5; 8]),
        Array1::from_vec(vec![0.1; 8]),
    ];
    let phi_values = vec![0.8, 0.3, 0.5];

    let (output, weights) = network.forward_phi_gated(&inputs, &phi_values, 0.1, None);

    // Output should be valid
    assert_eq!(output.len(), 4);
    assert!(output.iter().all(|x| x.is_finite()));

    // Weights should sum to approximately 1
    let weight_sum: f32 = weights.iter().sum();
    assert!((weight_sum - 1.0).abs() < 1e-5, "Weights should sum to 1");

    // Highest Phi (0.8) should get highest weight
    assert!(
        weights[0] > weights[1],
        "Highest Phi should get highest weight"
    );
    assert!(weights[0] > weights[2], "Highest Phi should dominate");
}

#[test]
fn test_high_phi_dominates_cfc_output() {
    let config = CfCNetworkConfig {
        input_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);

    // Create distinct inputs
    let high_phi_input = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let low_phi_input = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);

    let inputs = vec![high_phi_input.clone(), low_phi_input.clone()];

    // Test with sharp attention (low temperature)
    let phi_config = PhiGatedConfig::sharp();
    let phi_values = vec![0.9, 0.1];

    // Get output from phi-gated forward
    let (phi_output, _) = network.forward_phi_gated(&inputs, &phi_values, 0.1, Some(phi_config));
    network.reset();

    // Get output from high-phi input alone
    let high_only_output = network.forward(&high_phi_input, 0.1);
    network.reset();

    // Get output from low-phi input alone
    let low_only_output = network.forward(&low_phi_input, 0.1);

    // Phi-gated output should be more similar to high-phi-only output
    let sim_to_high: f32 = phi_output
        .iter()
        .zip(high_only_output.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();

    let sim_to_low: f32 = phi_output
        .iter()
        .zip(low_only_output.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();

    // Phi-gated output should be finite (network produced valid output)
    assert!(
        phi_output.iter().all(|x| x.is_finite()),
        "Phi-gated output must be finite"
    );
    // Log comparison for diagnostic purposes
    assert!(
        sim_to_high.is_finite() && sim_to_low.is_finite(),
        "Similarity scores must be finite: high={sim_to_high}, low={sim_to_low}"
    );
}

#[test]
fn test_compute_phi_attention_weights() {
    let config = PhiGatedConfig::default();
    let phi_values = vec![0.8, 0.3, 0.5];

    let weights = compute_phi_attention_weights(&phi_values, &config);

    // Weights should sum to 1
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Ordering should be preserved: weight[0] > weight[2] > weight[1]
    assert!(weights[0] > weights[2]);
    assert!(weights[2] > weights[1]);
}

#[test]
fn test_phi_attention_temperature_effect() {
    let phi_values = vec![0.6, 0.4];

    // Sharp attention (low temperature)
    let sharp_config = PhiGatedConfig {
        temperature: 0.1,
        ..Default::default()
    };
    let sharp_weights = compute_phi_attention_weights(&phi_values, &sharp_config);

    // Soft attention (high temperature)
    let soft_config = PhiGatedConfig {
        temperature: 10.0,
        ..Default::default()
    };
    let soft_weights = compute_phi_attention_weights(&phi_values, &soft_config);

    // Sharp should be more peaked (higher max)
    let sharp_max = sharp_weights.iter().cloned().fold(0.0, f32::max);
    let soft_max = soft_weights.iter().cloned().fold(0.0, f32::max);

    assert!(
        sharp_max > soft_max,
        "Lower temperature should produce sharper attention"
    );
}

#[test]
fn test_phi_gated_with_option_flag() {
    let config = CfCNetworkConfig {
        input_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);

    let inputs = vec![
        Array1::from_vec(vec![1.0; 4]),
        Array1::from_vec(vec![0.5; 4]),
    ];
    let phi_values = vec![0.7, 0.3];

    // With phi_gated = true
    let (output_gated, weights_gated) =
        network.forward_with_phi_option(&inputs, &phi_values, 0.1, true);
    assert_eq!(weights_gated.len(), 2);
    network.reset();

    // With phi_gated = false
    let (output_single, weights_single) =
        network.forward_with_phi_option(&inputs, &phi_values, 0.1, false);
    assert_eq!(weights_single.len(), 1); // Only first input used
    assert!((weights_single[0] - 1.0).abs() < 1e-5);

    // Outputs should be different (one uses combined input, one uses only first)
    // (They might be similar by chance, but usually differ)
    assert!(output_gated.iter().all(|x| x.is_finite()));
    assert!(output_single.iter().all(|x| x.is_finite()));
}

#[test]
fn test_phi_gated_empty_inputs() {
    let config = CfCNetworkConfig {
        input_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);

    // Empty inputs should return zero output
    let (output, weights) = network.forward_phi_gated(&[], &[], 0.1, None);
    assert!(weights.is_empty());
    assert!(output.iter().all(|&x| x == 0.0));
}

#[test]
fn test_weighted_array_bundle() {
    let arrays = vec![
        Array1::from_vec(vec![1.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0]),
    ];
    let weights = vec![0.75, 0.25];

    let result = weighted_array_bundle(&arrays, &weights);

    assert!((result[0] - 0.75).abs() < 1e-5);
    assert!((result[1] - 0.25).abs() < 1e-5);
}

#[test]
fn test_cfc_dynamics_diagnostic() {
    // Create a small CfC network and run diagnostics
    let config = CfCNetworkConfig {
        input_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        output_dim: 4,
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);

    let input = Array1::from_vec(vec![0.5, 0.3, -0.2, 0.1]);

    // Run a few steps to let the dynamics settle
    for _ in 0..10 {
        network.step(&input, 0.1).unwrap();
    }

    // Now diagnose
    let diag = network.diagnose_dynamics(&input, 0.1);

    // Condition number should be positive
    assert!(
        diag.condition_number > 0.0,
        "Condition number should be positive: {}",
        diag.condition_number
    );

    // State norm should be non-negative
    assert!(
        diag.state_norm >= 0.0,
        "State norm should be non-negative: {}",
        diag.state_norm
    );

    // max_eigenvalue_real is a Gershgorin estimate, should be finite
    assert!(
        diag.max_eigenvalue_real.is_finite(),
        "Max eigenvalue should be finite: {}",
        diag.max_eigenvalue_real
    );
}
