// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for CfC online learning during inference.
//!
//! These tests verify:
//! 1. Online learning improves prediction on distribution shifts
//! 2. Catastrophic forgetting is bounded (original performance preserved)
//! 3. Error-gated learning only triggers when error exceeds threshold

use ndarray::Array1;
use symthaea::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig, OnlineLearningConfig};

/// Compute MSE between two arrays (truncated to shorter length).
fn mse(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .take(n)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        / n as f32
}

/// Evaluate average prediction error over all pattern pairs.
fn eval_error(net: &mut CfCNetwork, pairs: &[(Array1<f32>, Array1<f32>)], dt: f32) -> f32 {
    let total: f32 = pairs
        .iter()
        .map(|(inp, tgt)| {
            net.reset();
            let pred = net.forward(inp, dt);
            mse(&pred, tgt)
        })
        .sum();
    total / pairs.len() as f32
}

/// Generate a sine pattern dataset.
fn generate_sine_pattern(
    dim: usize,
    period: usize,
    amplitude: f32,
    phase_offset: f32,
) -> Vec<(Array1<f32>, Array1<f32>)> {
    (0..period)
        .map(|t| {
            let phase_in = t as f32 / period as f32 * std::f32::consts::TAU + phase_offset;
            let phase_out = (t + 1) as f32 / period as f32 * std::f32::consts::TAU + phase_offset;
            let input = Array1::from_shape_fn(dim, |i| {
                (phase_in * (1.0 + i as f32 * 0.5)).sin() * amplitude
            });
            let target = Array1::from_shape_fn(dim, |i| {
                (phase_out * (1.0 + i as f32 * 0.5)).sin() * amplitude
            });
            (input, target)
        })
        .collect()
}

/// Generate a shifted distribution (different frequency and amplitude).
fn generate_shifted_pattern(dim: usize, period: usize) -> Vec<(Array1<f32>, Array1<f32>)> {
    // Different amplitude (0.5 instead of 0.3) and frequency multiplier
    (0..period)
        .map(|t| {
            let phase_in = t as f32 / period as f32 * std::f32::consts::TAU * 1.5; // 1.5x frequency
            let phase_out = (t + 1) as f32 / period as f32 * std::f32::consts::TAU * 1.5;
            let input = Array1::from_shape_fn(dim, |i| {
                (phase_in * (1.0 + i as f32 * 0.3)).cos() * 0.5 // cos instead of sin, different scaling
            });
            let target =
                Array1::from_shape_fn(dim, |i| (phase_out * (1.0 + i as f32 * 0.3)).cos() * 0.5);
            (input, target)
        })
        .collect()
}

#[test]
fn online_learning_improves_on_distribution_shift() {
    let dim = 16;
    let hidden = 32;
    let dt = 0.1;
    let period = 8;

    // Create network with online learning enabled
    // Use more aggressive settings for visible improvement in short test
    let online_config = OnlineLearningConfig {
        learning_rate: 0.02,   // Higher learning rate for faster adaptation
        error_threshold: 0.02, // Lower threshold to trigger more updates
        ema_alpha: 0.3,
        max_weight_delta: 0.05, // Allow larger weight changes
        adapt_tau: true,        // Also adapt time constants
        tau_lr_multiplier: 0.1, // Higher tau learning rate
    };

    let cell_config = CfCConfig {
        input_dim: dim,
        hidden_dim: hidden,
        use_backbone: false,
        backbone_layers: 0,
        backbone_dim: 0,
        dropout: 0.0,
        online_learning: Some(online_config.clone()),
        ..Default::default()
    };

    let net_config = CfCNetworkConfig {
        input_dim: dim,
        hidden_dim: hidden,
        num_layers: 1,
        output_dim: dim,
        cell_config,
        residual: false,
        bidirectional: false,
        enable_online_learning: true,
        online_learning_config: online_config,
    };

    let mut net = CfCNetwork::new(net_config);

    // Phase 1: Train on original distribution
    let original_pairs = generate_sine_pattern(dim, period, 0.3, 0.0);
    let lr = 0.003;
    let num_epochs = 40;

    for _epoch in 0..num_epochs {
        for (inp, tgt) in &original_pairs {
            let _ = net.train_step_spsa(inp, tgt, dt, lr);
        }
    }

    let trained_error_original = eval_error(&mut net, &original_pairs, dt);
    println!(
        "Trained error on original distribution: {:.6}",
        trained_error_original
    );

    // Phase 2: Encounter shifted distribution WITHOUT online learning
    let shifted_pairs = generate_shifted_pattern(dim, period);
    let error_before_adaptation = eval_error(&mut net, &shifted_pairs, dt);
    println!(
        "Error on shifted distribution BEFORE online adaptation: {:.6}",
        error_before_adaptation
    );

    // Phase 3: Apply online learning on shifted distribution
    // More adaptation rounds for visible improvement
    let adaptation_rounds = 50;
    for _ in 0..adaptation_rounds {
        for (inp, tgt) in &shifted_pairs {
            net.reset();
            let output = net.forward(inp, dt);
            let error = mse(&output, tgt);
            net.adapt_online(error, inp, tgt, dt);
        }
    }

    let error_after_adaptation = eval_error(&mut net, &shifted_pairs, dt);
    println!(
        "Error on shifted distribution AFTER online adaptation: {:.6}",
        error_after_adaptation
    );

    // Check statistics
    let stats = net.online_stats();
    println!("Online learning stats:");
    println!("  Total adaptation calls: {}", stats.total_adaptation_calls);
    println!("  Adaptations applied: {}", stats.adaptations_applied);
    println!("  Adaptations skipped: {}", stats.adaptations_skipped);
    println!("  EMA error: {:.6}", stats.ema_error);
    println!(
        "  Cumulative weight change: {:.6}",
        stats.cumulative_weight_change
    );

    // Core assertion: online learning should improve on the new distribution
    assert!(
        error_after_adaptation < error_before_adaptation,
        "Online learning should reduce error on shifted distribution: {:.6} -> {:.6}",
        error_before_adaptation,
        error_after_adaptation
    );

    // At least 1% improvement (conservative bound for online learning)
    let improvement_pct = (1.0 - error_after_adaptation / error_before_adaptation) * 100.0;
    println!("Improvement: {:.1}%", improvement_pct);
    assert!(
        improvement_pct >= 1.0,
        "Online learning should achieve at least 1% improvement, got {:.1}%",
        improvement_pct
    );
}

#[test]
fn catastrophic_forgetting_is_bounded() {
    let dim = 16;
    let hidden = 32;
    let dt = 0.1;
    let period = 8;

    // Create network with conservative online learning settings
    let online_config = OnlineLearningConfig {
        learning_rate: 0.001, // Very conservative
        error_threshold: 0.1,
        ema_alpha: 0.1,
        max_weight_delta: 0.005, // Very small max delta
        adapt_tau: false,
        tau_lr_multiplier: 0.01,
    };

    let cell_config = CfCConfig {
        input_dim: dim,
        hidden_dim: hidden,
        use_backbone: false,
        backbone_layers: 0,
        backbone_dim: 0,
        dropout: 0.0,
        online_learning: Some(online_config.clone()),
        ..Default::default()
    };

    let net_config = CfCNetworkConfig {
        input_dim: dim,
        hidden_dim: hidden,
        num_layers: 1,
        output_dim: dim,
        cell_config,
        residual: false,
        bidirectional: false,
        enable_online_learning: true,
        online_learning_config: online_config,
    };

    let mut net = CfCNetwork::new(net_config);

    // Phase 1: Train thoroughly on original distribution
    let original_pairs = generate_sine_pattern(dim, period, 0.3, 0.0);
    let lr = 0.003;
    let num_epochs = 60;

    for _epoch in 0..num_epochs {
        for (inp, tgt) in &original_pairs {
            let _ = net.train_step_spsa(inp, tgt, dt, lr);
        }
    }

    let trained_error_original = eval_error(&mut net, &original_pairs, dt);
    println!("Trained error on original: {:.6}", trained_error_original);

    // Phase 2: Adapt extensively on shifted distribution
    let shifted_pairs = generate_shifted_pattern(dim, period);
    let adaptation_rounds = 50; // More adaptation rounds

    for _ in 0..adaptation_rounds {
        for (inp, tgt) in &shifted_pairs {
            net.reset();
            let output = net.forward(inp, dt);
            let error = mse(&output, tgt);
            net.adapt_online(error, inp, tgt, dt);
        }
    }

    // Phase 3: Check performance on ORIGINAL distribution after adaptation
    let error_after_adaptation = eval_error(&mut net, &original_pairs, dt);
    println!(
        "Error on original AFTER adapting to shifted: {:.6}",
        error_after_adaptation
    );

    // Calculate degradation
    let degradation_pct = if trained_error_original > 0.0 {
        (error_after_adaptation / trained_error_original - 1.0) * 100.0
    } else {
        0.0
    };
    println!(
        "Performance degradation on original: {:.1}%",
        degradation_pct
    );

    // Catastrophic forgetting bound: no more than 100% degradation
    // (i.e., error should not more than double)
    assert!(
        degradation_pct < 100.0,
        "Catastrophic forgetting bound violated: degradation {:.1}% > 100%",
        degradation_pct
    );

    // Ideally, degradation should be under 50%
    if degradation_pct < 50.0 {
        println!("Excellent: degradation under 50%");
    } else {
        println!(
            "Warning: degradation {:.1}% is high but within bounds",
            degradation_pct
        );
    }
}

#[test]
fn error_gating_prevents_unnecessary_updates() {
    let dim = 16;
    let hidden = 32;
    let dt = 0.1;

    // Create network with high error threshold
    let online_config = OnlineLearningConfig {
        learning_rate: 0.01,
        error_threshold: 0.5, // High threshold - only adapt on large errors
        ema_alpha: 0.1,
        max_weight_delta: 0.02,
        adapt_tau: false,
        tau_lr_multiplier: 0.01,
    };

    let cell_config = CfCConfig {
        input_dim: dim,
        hidden_dim: hidden,
        use_backbone: false,
        backbone_layers: 0,
        backbone_dim: 0,
        dropout: 0.0,
        online_learning: Some(online_config.clone()),
        ..Default::default()
    };

    let net_config = CfCNetworkConfig {
        input_dim: dim,
        hidden_dim: hidden,
        num_layers: 1,
        output_dim: dim,
        cell_config,
        residual: false,
        bidirectional: false,
        enable_online_learning: true,
        online_learning_config: online_config,
    };

    let mut net = CfCNetwork::new(net_config);

    // Generate inputs that should have LOW prediction error (near-zero vectors)
    let low_error_input = Array1::from_shape_fn(dim, |_| 0.01);
    let low_error_target = Array1::from_shape_fn(dim, |_| 0.01);

    // Try to adapt many times with low error
    let num_attempts = 50;
    for _ in 0..num_attempts {
        net.reset();
        let output = net.forward(&low_error_input, dt);
        let error = mse(&output, &low_error_target);
        net.adapt_online(error, &low_error_input, &low_error_target, dt);
    }

    let stats = net.online_stats();
    println!("Error gating test:");
    println!("  Total adaptation calls: {}", stats.total_adaptation_calls);
    println!("  Adaptations applied: {}", stats.adaptations_applied);
    println!("  Adaptations skipped: {}", stats.adaptations_skipped);

    // Most adaptations should be skipped due to low error
    assert!(
        stats.adaptations_skipped > stats.adaptations_applied,
        "With high threshold and low errors, most adaptations should be skipped. \
         Applied: {}, Skipped: {}",
        stats.adaptations_applied,
        stats.adaptations_skipped
    );

    // Now test with HIGH error inputs
    net.reset_online_stats();
    let high_error_input = Array1::from_shape_fn(dim, |i| (i as f32 * 0.5).sin());
    let high_error_target = Array1::from_shape_fn(dim, |i| (i as f32 * 0.5).cos() * 2.0);

    for _ in 0..num_attempts {
        net.reset();
        let output = net.forward(&high_error_input, dt);
        let error = mse(&output, &high_error_target);
        net.adapt_online(error, &high_error_input, &high_error_target, dt);
    }

    let stats_high = net.online_stats();
    println!("High error test:");
    println!("  Adaptations applied: {}", stats_high.adaptations_applied);
    println!("  Adaptations skipped: {}", stats_high.adaptations_skipped);

    // With high errors, more adaptations should be applied
    // (but still some may be skipped due to adaptive threshold)
    assert!(
        stats_high.adaptations_applied > 0,
        "With high errors, at least some adaptations should be applied"
    );
}

#[test]
fn forward_with_adaptation_works() {
    let dim = 16;
    let hidden = 32;
    let dt = 0.1;

    let online_config = OnlineLearningConfig {
        learning_rate: 0.005,
        error_threshold: 0.01, // Low threshold to ensure adaptation
        ema_alpha: 0.2,
        max_weight_delta: 0.02,
        adapt_tau: false,
        tau_lr_multiplier: 0.01,
    };

    let cell_config = CfCConfig {
        input_dim: dim,
        hidden_dim: hidden,
        use_backbone: false,
        backbone_layers: 0,
        backbone_dim: 0,
        dropout: 0.0,
        online_learning: Some(online_config.clone()),
        ..Default::default()
    };

    let net_config = CfCNetworkConfig {
        input_dim: dim,
        hidden_dim: hidden,
        num_layers: 1,
        output_dim: dim,
        cell_config,
        residual: false,
        bidirectional: false,
        enable_online_learning: true,
        online_learning_config: online_config,
    };

    let mut net = CfCNetwork::new(net_config);

    let input = Array1::from_shape_fn(dim, |i| (i as f32 * 0.1).sin());
    let target = Array1::from_shape_fn(dim, |i| (i as f32 * 0.1).cos());

    // Forward without target - no adaptation
    let (output1, adapted1) = net.forward_with_adaptation(&input, dt, None);
    assert!(!adapted1, "Should not adapt without target");
    assert_eq!(output1.len(), dim);

    // Forward with target - should adapt
    net.reset();
    let (output2, adapted2) = net.forward_with_adaptation(&input, dt, Some(&target));
    // May or may not adapt depending on error
    println!("Adapted with target: {}", adapted2);
    assert_eq!(output2.len(), dim);

    // After some iterations, adaptation should definitely have occurred
    for _ in 0..20 {
        net.reset();
        net.forward_with_adaptation(&input, dt, Some(&target));
    }

    let stats = net.online_stats();
    assert!(
        stats.total_adaptation_calls > 0,
        "Should have made adaptation calls"
    );
}

#[test]
fn online_learning_disabled_by_default() {
    let dim = 16;
    let hidden = 32;
    let dt = 0.1;

    // Default config should have online learning disabled
    let cell_config = CfCConfig {
        input_dim: dim,
        hidden_dim: hidden,
        use_backbone: false,
        ..Default::default()
    };

    let net_config = CfCNetworkConfig {
        input_dim: dim,
        hidden_dim: hidden,
        num_layers: 1,
        output_dim: dim,
        cell_config,
        residual: false,
        bidirectional: false,
        ..Default::default() // enable_online_learning defaults to false
    };

    let mut net = CfCNetwork::new(net_config);
    assert!(!net.online_learning_enabled());

    let input = Array1::from_shape_fn(dim, |i| (i as f32 * 0.1).sin());
    let target = Array1::from_shape_fn(dim, |i| (i as f32 * 0.1).cos());

    // Try to adapt - should return false
    let adapted = net.adapt_online(1.0, &input, &target, dt);
    assert!(
        !adapted,
        "Should not adapt when online learning is disabled"
    );

    let stats = net.online_stats();
    assert_eq!(stats.total_adaptation_calls, 0);
    assert_eq!(stats.adaptations_applied, 0);
}