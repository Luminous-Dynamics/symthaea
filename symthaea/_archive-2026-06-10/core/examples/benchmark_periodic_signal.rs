// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Periodic Signal Learning Benchmark
//!
//! Tests CfC networks on periodic sequence prediction — the paper's #1 stated
//! limitation (Section 6.6): "On 4-element repeating sequences, prediction error
//! increases by 47.8% over 200 cycles."
//!
//! Root cause: local one-step optimization creates competing attractors.
//!
//! This benchmark validates two mitigation strategies:
//! 1. **Multi-scale loss**: weighted loss at 1, 2, 4, 8 steps ahead
//! 2. **Spectral regularization**: penalize mismatch in frequency domain
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_periodic_signal --release
//! ```

use std::f32::consts::PI;
use std::time::Instant;

use ndarray::Array1;
use symthaea::dynamics::cfc::{ActivationType, CfCConfig, CfCNetwork, CfCNetworkConfig};
use symthaea::dynamics::spectral_analysis::{SpectralAnalyzer, SpectralConfig, WindowType};
use symthaea::dynamics::stability_analysis::StabilityAnalyzer;

// ============================================================================
// Configuration
// ============================================================================

const INPUT_DIM: usize = 4;
const HIDDEN_DIM: usize = 32;
const OUTPUT_DIM: usize = 4;
const NUM_LAYERS: usize = 2;
const TRAIN_EPOCHS: usize = 50;
const LEARNING_RATE: f32 = 0.005;
const DT: f32 = 0.1;

// ============================================================================
// Periodic Sequence Generator
// ============================================================================

/// Generate a repeating pattern of given period using sin/cos phase encoding.
///
/// Each step i in [0..period) is encoded as a 4D vector:
///   [sin(2π·i/period), cos(2π·i/period), sin(4π·i/period), cos(4π·i/period)]
///
/// Returns (inputs, targets, dts) where target[t] = input[t+1].
fn generate_periodic_sequence(
    period: usize,
    total_steps: usize,
) -> (Vec<Array1<f32>>, Vec<Array1<f32>>, Vec<f32>) {
    let mut inputs = Vec::with_capacity(total_steps);
    let mut targets = Vec::with_capacity(total_steps);
    let dts = vec![DT; total_steps];

    for t in 0..total_steps {
        let phase = (t % period) as f32;
        let next_phase = ((t + 1) % period) as f32;
        let p = period as f32;

        let input = Array1::from_vec(vec![
            (2.0 * PI * phase / p).sin(),
            (2.0 * PI * phase / p).cos(),
            (4.0 * PI * phase / p).sin(),
            (4.0 * PI * phase / p).cos(),
        ]);

        let target = Array1::from_vec(vec![
            (2.0 * PI * next_phase / p).sin(),
            (2.0 * PI * next_phase / p).cos(),
            (4.0 * PI * next_phase / p).sin(),
            (4.0 * PI * next_phase / p).cos(),
        ]);

        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets, dts)
}

/// Compute per-cycle MSE at a given cycle horizon.
///
/// Runs the network forward for `cycles * period` steps, then measures
/// error in the last cycle.
fn measure_cycle_error(network: &mut CfCNetwork, period: usize, cycles: usize) -> f32 {
    let total_steps = cycles * period;
    let (inputs, targets, _) = generate_periodic_sequence(period, total_steps);

    network.reset();

    let mut last_cycle_mse = 0.0;
    let last_cycle_start = (cycles - 1) * period;

    for t in 0..total_steps {
        let output = network.forward(&inputs[t], DT);

        if t >= last_cycle_start {
            let diff = &output - &targets[t];
            last_cycle_mse += diff.iter().map(|x| x * x).sum::<f32>() / OUTPUT_DIM as f32;
        }
    }

    last_cycle_mse / period as f32
}

// ============================================================================
// Training Methods
// ============================================================================

/// Baseline training: standard BPTT on full sequences.
fn train_baseline(period: usize) -> CfCNetwork {
    let train_steps = period * 50; // 50 cycles of training data
    let (inputs, targets, dts) = generate_periodic_sequence(period, train_steps);

    let mut network = make_network();

    for epoch in 0..TRAIN_EPOCHS {
        let loss = network
            .train_step_bptt_optimized(&inputs, &targets, &dts, LEARNING_RATE)
            .unwrap_or(f32::MAX);

        if epoch % 10 == 0 {
            println!("    Baseline epoch {}: loss = {:.6}", epoch, loss);
        }
    }

    network
}

/// Multi-scale loss training: compute loss at multiple horizons.
///
/// L_total = 0.4*L1 + 0.3*L2 + 0.2*L4 + 0.1*L8
fn train_multiscale(period: usize) -> CfCNetwork {
    let train_steps = period * 50;
    let (inputs, targets, dts) = generate_periodic_sequence(period, train_steps);

    let mut network = make_network();

    // Multi-scale horizon weights
    let horizons: &[(usize, f32)] = &[
        (1, 0.4), // 1-step ahead (standard)
        (2, 0.3), // 2-step ahead
        (4, 0.2), // 4-step ahead
        (8, 0.1), // 8-step ahead
    ];

    for epoch in 0..TRAIN_EPOCHS {
        let mut total_loss = 0.0;

        for &(horizon, weight) in horizons {
            if horizon >= train_steps {
                continue;
            }

            // Create shifted targets: target[t] = input[t + horizon]
            let shifted_len = train_steps - horizon;
            let shifted_inputs = &inputs[..shifted_len];
            let shifted_targets: Vec<Array1<f32>> = (0..shifted_len)
                .map(|t| targets[(t + horizon - 1).min(train_steps - 1)].clone())
                .collect();
            let shifted_dts = &dts[..shifted_len];

            let loss = network
                .train_step_bptt_optimized(
                    shifted_inputs,
                    &shifted_targets,
                    shifted_dts,
                    LEARNING_RATE * weight,
                )
                .unwrap_or(f32::MAX);

            total_loss += weight * loss;
            network.reset();
        }

        if epoch % 10 == 0 {
            println!(
                "    Multiscale epoch {}: weighted loss = {:.6}",
                epoch, total_loss
            );
        }
    }

    network
}

/// Multi-scale + spectral regularization training.
///
/// Adds a spectral mismatch penalty: L += 0.1 * spectral_mse
fn train_multiscale_spectral(period: usize) -> CfCNetwork {
    let train_steps = period * 50;
    let (inputs, targets, dts) = generate_periodic_sequence(period, train_steps);

    let mut network = make_network();

    let spec_config = SpectralConfig {
        window_size: (period * 8).next_power_of_two().max(32),
        overlap: 0.5,
        window_type: WindowType::Hann,
        sample_rate: 1.0 / DT as f64,
    };
    let analyzer = SpectralAnalyzer::new(spec_config);

    let horizons: &[(usize, f32)] = &[(1, 0.4), (2, 0.3), (4, 0.2), (8, 0.1)];

    for epoch in 0..TRAIN_EPOCHS {
        let mut total_loss = 0.0;

        // Multi-scale training passes
        for &(horizon, weight) in horizons {
            if horizon >= train_steps {
                continue;
            }

            let shifted_len = train_steps - horizon;
            let shifted_inputs = &inputs[..shifted_len];
            let shifted_targets: Vec<Array1<f32>> = (0..shifted_len)
                .map(|t| targets[(t + horizon - 1).min(train_steps - 1)].clone())
                .collect();
            let shifted_dts = &dts[..shifted_len];

            let loss = network
                .train_step_bptt_optimized(
                    shifted_inputs,
                    &shifted_targets,
                    shifted_dts,
                    LEARNING_RATE * weight,
                )
                .unwrap_or(f32::MAX);

            total_loss += weight * loss;
            network.reset();
        }

        // Spectral regularization: compare predicted vs target spectra
        // Every 5 epochs to reduce overhead
        if epoch % 5 == 0 {
            let eval_steps = period * 10;
            let (eval_inputs, eval_targets, _) = generate_periodic_sequence(period, eval_steps);

            network.reset();
            let mut predicted_signal: Vec<f64> = Vec::with_capacity(eval_steps);
            let mut target_signal: Vec<f64> = Vec::with_capacity(eval_steps);

            for t in 0..eval_steps {
                let output = network.forward(&eval_inputs[t], DT);
                predicted_signal.push(output[0] as f64);
                target_signal.push(eval_targets[t][0] as f64);
            }

            let pred_psd = analyzer.psd(&predicted_signal);
            let tgt_psd = analyzer.psd(&target_signal);

            // Compute spectral MSE
            let spectral_len = pred_psd.psd.len().min(tgt_psd.psd.len());
            let spectral_mse: f64 = if spectral_len > 0 {
                pred_psd
                    .psd
                    .iter()
                    .zip(tgt_psd.psd.iter())
                    .take(spectral_len)
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / spectral_len as f64
            } else {
                0.0
            };

            // Use spectral mismatch as additional gradient signal:
            // Train one more step with spectral-weighted learning rate
            let spectral_weight = 0.1 * (spectral_mse as f32).min(10.0);
            if spectral_weight > 0.001 {
                let _ = network.train_step_bptt_optimized(
                    &inputs,
                    &targets,
                    &dts,
                    LEARNING_RATE * spectral_weight,
                );
                network.reset();
            }

            total_loss += spectral_mse as f32 * 0.1;
        }

        if epoch % 10 == 0 {
            println!("    Spectral epoch {}: loss = {:.6}", epoch, total_loss);
        }
    }

    network
}

// ============================================================================
// Analysis
// ============================================================================

/// Find dominant frequency in output via FFT.
fn dominant_frequency(network: &mut CfCNetwork, period: usize) -> f64 {
    let eval_steps = period * 20;
    let (eval_inputs, _, _) = generate_periodic_sequence(period, eval_steps);

    let spec_config = SpectralConfig {
        window_size: eval_steps.next_power_of_two().max(32),
        overlap: 0.0,
        window_type: WindowType::Hann,
        sample_rate: 1.0 / DT as f64,
    };
    let analyzer = SpectralAnalyzer::new(spec_config);

    network.reset();
    let mut signal: Vec<f64> = Vec::with_capacity(eval_steps);
    for eval_input in eval_inputs.iter().take(eval_steps) {
        let output = network.forward(eval_input, DT);
        signal.push(output[0] as f64);
    }

    let psd = analyzer.psd(&signal);
    // Find peak frequency (skip DC at index 0)
    if psd.psd.len() > 1 {
        let (peak_idx, _) = psd
            .psd
            .iter()
            .enumerate()
            .skip(1) // skip DC
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        if peak_idx < psd.frequencies.len() {
            psd.frequencies[peak_idx]
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Compute Lyapunov exponent of CfC hidden state trajectory.
fn analyze_stability(network: &mut CfCNetwork, period: usize) -> f64 {
    let stability = StabilityAnalyzer::with_defaults(HIDDEN_DIM);

    // Build a dynamics function from the CfC network's hidden state evolution
    let eval_steps = period * 20;
    let (eval_inputs, _, _) = generate_periodic_sequence(period, eval_steps);

    network.reset();

    // Collect hidden state trajectory
    let mut states: Vec<Vec<f64>> = Vec::new();
    for eval_input in eval_inputs.iter().take(eval_steps) {
        let _ = network.forward(eval_input, DT);
        if let Ok(state) = network.read_state() {
            states.push(state.iter().map(|&x| x as f64).collect());
        }
    }

    if states.len() < 10 {
        return 0.0;
    }

    // Approximate dynamics from trajectory differences
    let x0 = states[0].clone();
    let dynamics = move |x: &[f64]| -> Vec<f64> {
        // Find closest state in trajectory and use next state as dynamics estimate
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, s) in states.iter().enumerate().take(states.len() - 1) {
            let dist: f64 = s.iter().zip(x.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        // Return approximate derivative
        states[best_idx + 1]
            .iter()
            .zip(states[best_idx].iter())
            .map(|(next, curr)| (next - curr) / DT as f64)
            .collect()
    };

    let lyap = stability.lyapunov_exponents(&dynamics, &x0, DT as f64, 500);

    lyap.max_exponent
}

// ============================================================================
// Helpers
// ============================================================================

fn make_network() -> CfCNetwork {
    let cell_config = CfCConfig {
        input_dim: INPUT_DIM,
        hidden_dim: HIDDEN_DIM,
        use_backbone: true,
        backbone_layers: 1,
        backbone_dim: 16,
        activation: ActivationType::SiLU,
        tau_range: (0.1, 5.0), // Wider range for periodic dynamics
        dropout: 0.0,
        gradient_clip: 5.0,
        online_learning: None,
    };

    let config = CfCNetworkConfig {
        input_dim: INPUT_DIM,
        hidden_dim: HIDDEN_DIM,
        num_layers: NUM_LAYERS,
        output_dim: OUTPUT_DIM,
        cell_config,
        residual: true,
        bidirectional: false,
        enable_online_learning: false,
        online_learning_config: Default::default(),
    };

    CfCNetwork::new(config)
}

// ============================================================================
// Results
// ============================================================================

#[derive(Debug)]
struct PeriodResult {
    period: usize,
    baseline_err_50: f32,
    baseline_err_200: f32,
    baseline_growth: f32,
    multiscale_err_50: f32,
    multiscale_err_200: f32,
    multiscale_growth: f32,
    spectral_err_50: f32,
    spectral_err_200: f32,
    spectral_growth: f32,
    dominant_freq_baseline: f64,
    dominant_freq_spectral: f64,
    expected_freq: f64,
    lyapunov_max: f64,
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    Periodic Signal Learning Benchmark                        ║");
    println!("║    CfC multi-scale + spectral regularization                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let periods = [2, 4, 8, 16];
    let mut results: Vec<PeriodResult> = Vec::new();
    let total_start = Instant::now();

    for &period in &periods {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "Period: {} (expected freq: {:.3} Hz)",
            period,
            1.0 / (period as f64 * DT as f64)
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let expected_freq = 1.0 / (period as f64 * DT as f64);

        // 1. Baseline training
        println!("  [1/3] Baseline BPTT training...");
        let start = Instant::now();
        let mut baseline = train_baseline(period);
        println!("    Trained in {:.1}s", start.elapsed().as_secs_f64());

        let baseline_err_50 = measure_cycle_error(&mut baseline, period, 50);
        let baseline_err_200 = measure_cycle_error(&mut baseline, period, 200);
        let baseline_growth = if baseline_err_50 > 1e-10 {
            (baseline_err_200 - baseline_err_50) / baseline_err_50 * 100.0
        } else {
            0.0
        };
        println!(
            "    Error@50c: {:.6}, Error@200c: {:.6}, Growth: {:.1}%",
            baseline_err_50, baseline_err_200, baseline_growth
        );

        let dom_freq_baseline = dominant_frequency(&mut baseline, period);
        println!(
            "    Dominant freq: {:.3} Hz (expected: {:.3})",
            dom_freq_baseline, expected_freq
        );

        // 2. Multi-scale training
        println!("  [2/3] Multi-scale loss training...");
        let start = Instant::now();
        let mut multiscale = train_multiscale(period);
        println!("    Trained in {:.1}s", start.elapsed().as_secs_f64());

        let multiscale_err_50 = measure_cycle_error(&mut multiscale, period, 50);
        let multiscale_err_200 = measure_cycle_error(&mut multiscale, period, 200);
        let multiscale_growth = if multiscale_err_50 > 1e-10 {
            (multiscale_err_200 - multiscale_err_50) / multiscale_err_50 * 100.0
        } else {
            0.0
        };
        println!(
            "    Error@50c: {:.6}, Error@200c: {:.6}, Growth: {:.1}%",
            multiscale_err_50, multiscale_err_200, multiscale_growth
        );

        // 3. Multi-scale + spectral
        println!("  [3/3] Multi-scale + spectral regularization...");
        let start = Instant::now();
        let mut spectral = train_multiscale_spectral(period);
        println!("    Trained in {:.1}s", start.elapsed().as_secs_f64());

        let spectral_err_50 = measure_cycle_error(&mut spectral, period, 50);
        let spectral_err_200 = measure_cycle_error(&mut spectral, period, 200);
        let spectral_growth = if spectral_err_50 > 1e-10 {
            (spectral_err_200 - spectral_err_50) / spectral_err_50 * 100.0
        } else {
            0.0
        };
        println!(
            "    Error@50c: {:.6}, Error@200c: {:.6}, Growth: {:.1}%",
            spectral_err_50, spectral_err_200, spectral_growth
        );

        let dom_freq_spectral = dominant_frequency(&mut spectral, period);
        println!(
            "    Dominant freq: {:.3} Hz (expected: {:.3})",
            dom_freq_spectral, expected_freq
        );

        // 4. Stability analysis on best model
        println!("  Lyapunov analysis...");
        let lyap = analyze_stability(&mut spectral, period);
        println!(
            "    Max Lyapunov exponent: {:.4} ({})",
            lyap,
            if lyap < 0.0 {
                "stable attractor"
            } else if lyap > 0.01 {
                "chaotic"
            } else {
                "neutral"
            }
        );

        results.push(PeriodResult {
            period,
            baseline_err_50,
            baseline_err_200,
            baseline_growth,
            multiscale_err_50,
            multiscale_err_200,
            multiscale_growth,
            spectral_err_50,
            spectral_err_200,
            spectral_growth,
            dominant_freq_baseline: dom_freq_baseline,
            dominant_freq_spectral: dom_freq_spectral,
            expected_freq,
            lyapunov_max: lyap,
        });

        println!();
    }

    // ========================================================================
    // Summary Table
    // ========================================================================

    let total_duration = total_start.elapsed();

    println!("╔═════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PERIODIC SIGNAL BENCHMARK SUMMARY                            ║");
    println!("╠═══════╤══════════════════╤══════════════════╤══════════════════╤════════════════╣");
    println!("║ Period│ Baseline Growth  │ Multiscale Growth│ Spectral Growth  │ Dom.Freq Match ║");
    println!("╟───────┼──────────────────┼──────────────────┼──────────────────┼────────────────╢");

    for r in &results {
        let freq_match =
            (r.dominant_freq_spectral - r.expected_freq).abs() / r.expected_freq < 0.15;
        println!(
            "║   {:3} │ {:+7.1}%         │ {:+7.1}%         │ {:+7.1}%         │ {}             ║",
            r.period,
            r.baseline_growth,
            r.multiscale_growth,
            r.spectral_growth,
            if freq_match { "PASS" } else { "FAIL" },
        );
    }
    println!("╚═══════╧══════════════════╧══════════════════╧══════════════════╧════════════════╝");

    // ========================================================================
    // Pass/Fail Criteria
    // ========================================================================

    println!("\n--- Pass/Fail Criteria ---");
    let mut pass_count = 0;
    let mut total_tests = 0;

    for r in &results {
        // Test 1: Period-2 should have near-zero error
        if r.period == 2 {
            total_tests += 1;
            let pass = r.spectral_err_200 < 0.05;
            if pass {
                pass_count += 1;
            }
            println!(
                "  [{}] Period-2 error < 0.05: {:.6}",
                if pass { "PASS" } else { "FAIL" },
                r.spectral_err_200
            );
        }

        // Test 2: Period-4 baseline reproduces known issue (~47% growth)
        if r.period == 4 {
            total_tests += 1;
            let pass = r.baseline_growth > 10.0; // At least 10% growth (may not hit 47% with only 50 epochs)
            if pass {
                pass_count += 1;
            }
            println!(
                "  [{}] Period-4 baseline growth > 10%: {:.1}%",
                if pass { "PASS" } else { "FAIL" },
                r.baseline_growth
            );
        }

        // Test 3: Multi-scale reduces growth vs baseline
        if r.period == 4 {
            total_tests += 1;
            let pass = r.multiscale_growth < r.baseline_growth;
            if pass {
                pass_count += 1;
            }
            println!(
                "  [{}] Period-4 multiscale < baseline: {:.1}% < {:.1}%",
                if pass { "PASS" } else { "FAIL" },
                r.multiscale_growth,
                r.baseline_growth
            );
        }

        // Test 4: Spectral further reduces growth
        if r.period == 4 {
            total_tests += 1;
            let pass = r.spectral_growth < r.multiscale_growth;
            if pass {
                pass_count += 1;
            }
            println!(
                "  [{}] Period-4 spectral < multiscale: {:.1}% < {:.1}%",
                if pass { "PASS" } else { "FAIL" },
                r.spectral_growth,
                r.multiscale_growth
            );
        }

        // Test 5: Dominant frequency matches input period
        total_tests += 1;
        let freq_error = (r.dominant_freq_spectral - r.expected_freq).abs() / r.expected_freq;
        let pass = freq_error < 0.15;
        if pass {
            pass_count += 1;
        }
        println!(
            "  [{}] Period-{} freq match (err={:.1}%): {:.3} vs {:.3} Hz",
            if pass { "PASS" } else { "FAIL" },
            r.period,
            freq_error * 100.0,
            r.dominant_freq_spectral,
            r.expected_freq
        );
    }

    println!("\n  Score: {}/{} tests passed", pass_count, total_tests);
    println!("  Duration: {:.1}s", total_duration.as_secs_f64());

    // ========================================================================
    // Save Results
    // ========================================================================

    let output_dir = std::path::Path::new("data/benchmarks/periodic_signal");
    if std::fs::create_dir_all(output_dir).is_ok() {
        let results_json = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "config": {
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "output_dim": OUTPUT_DIM,
                "num_layers": NUM_LAYERS,
                "train_epochs": TRAIN_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "dt": DT,
            },
            "results": results.iter().map(|r| serde_json::json!({
                "period": r.period,
                "expected_freq_hz": r.expected_freq,
                "baseline": {
                    "error_50c": r.baseline_err_50,
                    "error_200c": r.baseline_err_200,
                    "growth_pct": r.baseline_growth,
                    "dominant_freq_hz": r.dominant_freq_baseline,
                },
                "multiscale": {
                    "error_50c": r.multiscale_err_50,
                    "error_200c": r.multiscale_err_200,
                    "growth_pct": r.multiscale_growth,
                },
                "spectral": {
                    "error_50c": r.spectral_err_50,
                    "error_200c": r.spectral_err_200,
                    "growth_pct": r.spectral_growth,
                    "dominant_freq_hz": r.dominant_freq_spectral,
                },
                "lyapunov_max": r.lyapunov_max,
            })).collect::<Vec<_>>(),
            "summary": {
                "pass_count": pass_count,
                "total_tests": total_tests,
                "duration_secs": total_duration.as_secs_f64(),
            },
        });

        let output_path = output_dir.join("results.json");
        if let Ok(file) = std::fs::File::create(&output_path) {
            if serde_json::to_writer_pretty(file, &results_json).is_ok() {
                println!("\n  Results saved to {}", output_path.display());
            }
        }
    }
}