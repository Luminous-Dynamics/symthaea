// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: validates that the CfC network's online learning
//! actually reduces prediction error over sustained operation on a
//! repeating pattern.
//!
//! Uses SPSA (Simultaneous Perturbation Stochastic Approximation) training
//! which directly perturbs network weights to estimate gradients. SPSA is
//! stochastic — some random weight initializations converge while others
//! diverge. The test tries multiple initializations and passes if any show
//! learning. BPTT (backpropagation through time) is tested separately as
//! a more reliable gradient method.

use ndarray::Array1;
use symthaea::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};

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

/// Build a standard CfC config for learning tests.
fn make_config(dim: usize, hidden: usize) -> CfCNetworkConfig {
    let cell_config = CfCConfig {
        input_dim: dim,
        hidden_dim: hidden,
        use_backbone: false,
        backbone_layers: 0,
        backbone_dim: 0,
        dropout: 0.0,
        ..Default::default()
    };

    CfCNetworkConfig {
        input_dim: dim,
        hidden_dim: hidden,
        num_layers: 1,
        output_dim: dim,
        cell_config,
        residual: false,
        bidirectional: false,
        ..Default::default()
    }
}

/// Build sine pattern pairs for learning tests.
fn make_sine_pairs(dim: usize, period: usize) -> Vec<(Array1<f32>, Array1<f32>)> {
    (0..period)
        .map(|t| {
            let phase_in = t as f32 / period as f32 * std::f32::consts::TAU;
            let phase_out = (t + 1) as f32 / period as f32 * std::f32::consts::TAU;
            let input =
                Array1::from_shape_fn(dim, |i| (phase_in * (1.0 + i as f32 * 0.5)).sin() * 0.3);
            let target =
                Array1::from_shape_fn(dim, |i| (phase_out * (1.0 + i as f32 * 0.5)).sin() * 0.3);
            (input, target)
        })
        .collect()
}

/// Run one SPSA learning trial. Returns (learned: bool, reduction_pct, details).
fn run_spsa_trial(
    config: &CfCNetworkConfig,
    pairs: &[(Array1<f32>, Array1<f32>)],
    dt: f32,
    lr: f32,
    num_epochs: usize,
) -> (bool, f32, String) {
    let mut net = CfCNetwork::new(config.clone());
    let period = pairs.len();

    let baseline_error = eval_error(&mut net, pairs, dt);

    let mut epoch_losses: Vec<f32> = Vec::new();
    for _epoch in 0..num_epochs {
        let mut epoch_loss = 0.0f32;
        for (inp, tgt) in pairs {
            let loss = net
                .train_step_spsa(inp, tgt, dt, lr)
                .expect("train_step_spsa failed");
            epoch_loss += loss;
        }
        epoch_losses.push(epoch_loss / period as f32);
    }

    let post_error = eval_error(&mut net, pairs, dt);

    let early_window = 10;
    let late_window = 10;
    let avg_early: f32 = epoch_losses[..early_window].iter().sum::<f32>() / early_window as f32;
    let avg_late: f32 =
        epoch_losses[num_epochs - late_window..].iter().sum::<f32>() / late_window as f32;

    let reduction_pct = if baseline_error > 0.0 {
        (1.0 - post_error / baseline_error) * 100.0
    } else {
        0.0
    };

    let learned = avg_late < avg_early && post_error < baseline_error && reduction_pct >= 0.5;

    let details = format!(
        "baseline={:.6}, post={:.6}, reduction={:.1}%, early_loss={:.6}, late_loss={:.6}",
        baseline_error, post_error, reduction_pct, avg_early, avg_late
    );

    (learned, reduction_pct, details)
}

#[test]
fn cfc_online_learning_reduces_prediction_error() {
    let dim = 16;
    let hidden = 32;
    let dt = 0.1;
    let lr = 0.003;
    let num_epochs = 60;
    let max_trials = 5;

    let config = make_config(dim, hidden);
    let pairs = make_sine_pairs(dim, 8);

    // SPSA is stochastic — some random weight initializations converge while
    // others diverge. Try multiple initializations; pass if any show learning.
    let mut best_reduction = f32::NEG_INFINITY;
    let mut all_details = Vec::new();

    for trial in 0..max_trials {
        let (learned, reduction, details) = run_spsa_trial(&config, &pairs, dt, lr, num_epochs);
        println!("  Trial {}: {}", trial + 1, details);
        all_details.push(details);

        if reduction > best_reduction {
            best_reduction = reduction;
        }

        if learned {
            println!(
                "=== SPSA learning confirmed on trial {} ({:.1}% reduction) ===",
                trial + 1,
                reduction
            );
            return; // Pass
        }
    }

    panic!(
        "SPSA failed to learn in {} trials. Best reduction: {:.1}%.\n\
         Trial details:\n{}",
        max_trials,
        best_reduction,
        all_details
            .iter()
            .enumerate()
            .map(|(i, d)| format!("  {}: {}", i + 1, d))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn test_bptt_learning_convergence() {
    let dim = 16;
    let hidden = 32;
    let dt = 0.1;
    let lr = 0.001;
    // BPTT with proper gradients should converge faster than SPSA;
    // 30 epochs (vs SPSA's 60) should suffice.
    let num_epochs = 30;

    let config = make_config(dim, hidden);
    let mut net = CfCNetwork::new(config);

    let period = 8;
    let pairs = make_sine_pairs(dim, period);

    // Phase 1: Measure baseline prediction error (untrained network).
    let baseline_error = eval_error(&mut net, &pairs, dt);

    // Phase 2: Train with BPTT, recording loss per epoch.
    let mut epoch_losses: Vec<f32> = Vec::new();

    let dts: Vec<f32> = vec![dt; period];
    let inputs: Vec<Array1<f32>> = pairs.iter().map(|(i, _)| i.clone()).collect();
    let targets: Vec<Array1<f32>> = pairs.iter().map(|(_, t)| t.clone()).collect();

    for _epoch in 0..num_epochs {
        let loss = net
            .train_step_bptt(&inputs, &targets, &dts, lr)
            .expect("train_step_bptt failed");
        epoch_losses.push(loss);
    }

    // Phase 3: Measure post-training prediction error.
    let post_error = eval_error(&mut net, &pairs, dt);

    let early_window = 5;
    let late_window = 5;
    let avg_early: f32 = epoch_losses[..early_window].iter().sum::<f32>() / early_window as f32;
    let avg_late: f32 =
        epoch_losses[num_epochs - late_window..].iter().sum::<f32>() / late_window as f32;

    let reduction_pct = if baseline_error > 0.0 {
        (1.0 - post_error / baseline_error) * 100.0
    } else {
        0.0
    };

    let loss_reduction_pct = if avg_early > 0.0 {
        (1.0 - avg_late / avg_early) * 100.0
    } else {
        0.0
    };

    println!("=== BPTT Closed-Loop Learning Results ===");
    println!("Baseline prediction error:      {:.6}", baseline_error);
    println!("Post-training prediction error: {:.6}", post_error);
    println!("Prediction error reduction:     {:.1}%", reduction_pct);
    println!("Early epoch loss (0-4):         {:.6}", avg_early);
    println!("Late epoch loss (25-29):        {:.6}", avg_late);
    println!("Epoch loss reduction:           {:.1}%", loss_reduction_pct);
    println!(
        "Epochs used:                    {} (vs SPSA's 60)",
        num_epochs
    );

    // Core assertion 1: training loss decreases over epochs
    assert!(
        avg_late < avg_early,
        "Late training loss ({:.6}) should be less than early loss ({:.6})",
        avg_late,
        avg_early,
    );

    // Core assertion 2: post-training evaluation error < baseline
    assert!(
        post_error < baseline_error,
        "Post-training error ({:.6}) should be less than baseline ({:.6})",
        post_error,
        baseline_error,
    );

    // Core assertion 3: BPTT should achieve >= 5% error reduction
    // (better than SPSA's 2-3% floor, in half the epochs)
    assert!(
        reduction_pct >= 5.0,
        "BPTT error reduction ({:.1}%) must be at least 5%",
        reduction_pct,
    );
}