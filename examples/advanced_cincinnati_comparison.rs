// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced Cincinnati-LTC Comparison
//!
//! Compares baseline, enhanced, and advanced implementations:
//! - Baseline: Original Cincinnati-LTC
//! - Enhanced: Multi-scale + amplitude encoding + attention
//! - Advanced: + Chaos detection + adaptive weights + memory horizon
//!
//! Run with: cargo run --example advanced_cincinnati_comparison

use std::f64::consts::PI;
use symthaea::hdc::cincinnati_advanced::AdvancedCincinnatiEngine;
use symthaea::hdc::cincinnati_enhanced::EnhancedCincinnatiEngine;
use symthaea::hdc::cincinnati_ltc::CincinnatiLtcEngine;
use symthaea::hdc::unified_hv::ContinuousHV;

// =============================================================================
// SIGNAL GENERATORS
// =============================================================================

/// Generate EEG-like alpha rhythm (8-12 Hz)
fn generate_alpha(sample_rate: f64, duration: f64) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let freq = 10.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let base = (2.0 * PI * freq * t).sin();
            let harmonic = 0.3 * (2.0 * PI * freq * 2.0 * t).sin();
            let noise = 0.2 * ((t * 12345.67).sin() * (t * 7654.32).cos());
            base + harmonic + noise
        })
        .collect()
}

/// Generate EEG-like beta rhythm (12-30 Hz)
fn generate_beta(sample_rate: f64, duration: f64) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let freq = 20.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let base = (2.0 * PI * freq * t).sin();
            let modulation = 1.0 + 0.3 * (2.0 * PI * 0.5 * t).sin();
            base * modulation
        })
        .collect()
}

/// Generate EEG-like gamma rhythm (30-100 Hz)
fn generate_gamma(sample_rate: f64, duration: f64) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let freq = 40.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let gamma = (2.0 * PI * freq * t).sin();
            let theta_envelope = (2.0 * PI * 6.0 * t).sin().max(0.0);
            gamma * (0.3 + 0.7 * theta_envelope)
        })
        .collect()
}

/// Generate HRV pattern (1.2 Hz base with respiratory modulation)
fn generate_hrv(sample_rate: f64, duration: f64) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let base_rate = 1.2;
    let respiratory_mod = 0.15;

    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let resp_mod = respiratory_mod * (2.0 * PI * 0.25 * t).sin();
            let instant_rate = base_rate * (1.0 + resp_mod);
            let phase = (2.0 * PI * instant_rate * t) % (2.0 * PI);

            if phase < 0.1 * PI {
                (phase / (0.1 * PI) * PI).sin()
            } else {
                0.0
            }
        })
        .collect()
}

/// Generate respiratory pattern (0.25 Hz)
fn generate_respiratory(sample_rate: f64, duration: f64) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let breath_rate = 0.25;

    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let phase = (2.0 * PI * breath_rate * t) % (2.0 * PI);

            let breath = if phase < PI * 1.2 {
                (phase / 1.2).sin()
            } else {
                -((phase - PI * 1.2) / 0.8 * PI).sin()
            };

            breath + 0.05 * (t * 0.1).sin()
        })
        .collect()
}

/// Generate square wave
fn generate_square_wave(sample_rate: f64, duration: f64, half_period: usize) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    (0..n)
        .map(|i| {
            if (i / half_period).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Generate logistic map (r=3.2 is periodic, r=3.8 is chaotic)
fn generate_logistic(r: f64, length: usize) -> Vec<f64> {
    let mut x = 0.5;
    (0..length)
        .map(|_| {
            x = r * x * (1.0 - x);
            (x - 0.5) * 2.0
        })
        .collect()
}

/// Generate Henon map (chaotic attractor)
fn generate_henon(length: usize) -> Vec<f64> {
    let a = 1.4;
    let b = 0.3;
    let mut x = 0.1;
    let mut y = 0.1;

    (0..length)
        .map(|_| {
            let x_new = 1.0 - a * x * x + y;
            let y_new = b * x;
            x = x_new;
            y = y_new;
            x
        })
        .collect()
}

/// Generate Lorenz system (chaotic)
fn generate_lorenz(length: usize, dt: f64) -> Vec<f64> {
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;

    let mut x = 1.0;
    let mut y = 1.0;
    let mut z = 1.0;

    (0..length)
        .map(|_| {
            let dx = sigma * (y - x);
            let dy = x * (rho - z) - y;
            let dz = x * y - beta * z;

            x += dx * dt;
            y += dy * dt;
            z += dz * dt;

            x / 20.0 // Normalize
        })
        .collect()
}

// =============================================================================
// TEST FUNCTIONS
// =============================================================================

fn test_baseline(signal: &[f64], _name: &str) -> f64 {
    let threshold = 0.0;
    let mut engine = CincinnatiLtcEngine::new(5);
    engine.set_budding_threshold(0.5);
    engine.set_sustain_steps(3);

    let mut correct = 0;
    let mut total = 0;

    let mut node_states: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(1024, i as u64 * 1000))
        .collect();

    for (i, &sample) in signal.iter().enumerate() {
        let above = sample > threshold;

        if i >= 50 {
            let (pred, _) = engine.predict();
            if pred == above {
                correct += 1;
            }
            total += 1;
        }

        let input = ContinuousHV::random(1024, i as u64);
        engine.step(above, &input);

        let node_count = engine.node_count();
        while node_states.len() < node_count {
            node_states.push(ContinuousHV::random(
                1024,
                (node_states.len() * 1000 + i) as u64,
            ));
        }

        for node_id in 0..node_count {
            let expected = ContinuousHV::random(1024, if above { 111111 } else { 222222 });
            let actual = ContinuousHV::random(1024, if above { 111111 } else { 222222 });
            engine.update_prediction_error(node_id, &expected, &actual);
        }
        let _ = engine.process_budding(&node_states[..node_count], i as f64);
    }

    if total > 0 {
        correct as f64 / total as f64
    } else {
        0.5
    }
}

fn test_enhanced(signal: &[f64], sample_rate: f32, _name: &str) -> f64 {
    let mut engine = EnhancedCincinnatiEngine::new(sample_rate);

    for &sample in signal.iter() {
        engine.process_signal(sample);
    }

    engine.stats().accuracy as f64
}

fn test_advanced(signal: &[f64], sample_rate: f32, _name: &str) -> (f64, bool, f64, [f32; 3]) {
    let mut engine = AdvancedCincinnatiEngine::new(sample_rate);

    for &sample in signal.iter() {
        engine.process(sample);
    }

    let stats = engine.stats();
    (
        stats.accuracy as f64,
        stats.chaos_metrics.is_chaotic,
        stats.chaos_metrics.lyapunov_exponent,
        stats.weights,
    )
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!(
        "
╔══════════════════════════════════════════════════════════════════════════════╗
║              ADVANCED CINCINNATI-LTC COMPARISON                              ║
║                                                                              ║
║  Comparing three implementation levels:                                      ║
║    1. Baseline: Original Cincinnati-LTC                                      ║
║    2. Enhanced: + Multi-scale + amplitude + attention                        ║
║    3. Advanced: + Chaos detection + adaptive weights + memory horizon        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"
    );

    let sample_rate = 250.0;
    let duration = 4.0;

    // Standard test signals
    let signals: Vec<(&str, Vec<f64>, bool)> = vec![
        (
            "EEG Alpha (10 Hz)",
            generate_alpha(sample_rate, duration),
            false,
        ),
        (
            "EEG Beta (20 Hz)",
            generate_beta(sample_rate, duration),
            false,
        ),
        (
            "EEG Gamma (40 Hz)",
            generate_gamma(sample_rate, duration),
            false,
        ),
        ("HRV (1.2 Hz)", generate_hrv(sample_rate, duration), false),
        (
            "Respiratory",
            generate_respiratory(sample_rate, duration),
            false,
        ),
        (
            "Square Wave (p=8)",
            generate_square_wave(sample_rate, duration, 4),
            false,
        ),
        ("Logistic r=3.2", generate_logistic(3.2, 1000), false),
        ("Logistic r=3.8", generate_logistic(3.8, 1000), true), // Chaotic!
        ("Henon Map", generate_henon(1000), true),              // Chaotic!
        ("Lorenz System", generate_lorenz(1000, 0.01), true),   // Chaotic!
    ];

    // Header
    println!("\n{:=^90}", " ACCURACY COMPARISON ");
    println!(
        "\n{:<20} │ {:>10} │ {:>10} │ {:>10} │ {:>8} │ {:>8}",
        "Signal", "Baseline", "Enhanced", "Advanced", "Δ Enh", "Δ Adv"
    );
    println!(
        "{:─<20}─┼─{:─^10}─┼─{:─^10}─┼─{:─^10}─┼─{:─^8}─┼─{:─^8}",
        "", "", "", "", "", ""
    );

    let mut totals = (0.0, 0.0, 0.0);
    let mut count = 0;

    for (name, signal, _expected_chaotic) in &signals {
        let baseline = test_baseline(signal, name);
        let enhanced = test_enhanced(signal, sample_rate as f32, name);
        let (advanced, _is_chaotic, _lyap, _weights) =
            test_advanced(signal, sample_rate as f32, name);

        let delta_enh = (enhanced - baseline) * 100.0;
        let delta_adv = (advanced - baseline) * 100.0;

        println!(
            "{:<20} │ {:>9.1}% │ {:>9.1}% │ {:>9.1}% │ {:>+7.1}% │ {:>+7.1}%",
            name,
            baseline * 100.0,
            enhanced * 100.0,
            advanced * 100.0,
            delta_enh,
            delta_adv
        );

        totals.0 += baseline;
        totals.1 += enhanced;
        totals.2 += advanced;
        count += 1;
    }

    println!(
        "{:─<20}─┼─{:─^10}─┼─{:─^10}─┼─{:─^10}─┼─{:─^8}─┼─{:─^8}",
        "", "", "", "", "", ""
    );

    let avg_baseline = totals.0 / count as f64;
    let avg_enhanced = totals.1 / count as f64;
    let avg_advanced = totals.2 / count as f64;

    println!(
        "{:<20} │ {:>9.1}% │ {:>9.1}% │ {:>9.1}% │ {:>+7.1}% │ {:>+7.1}%",
        "AVERAGE",
        avg_baseline * 100.0,
        avg_enhanced * 100.0,
        avg_advanced * 100.0,
        (avg_enhanced - avg_baseline) * 100.0,
        (avg_advanced - avg_baseline) * 100.0
    );

    // Chaos detection results
    println!("\n{:=^90}", " CHAOS DETECTION ANALYSIS ");
    println!(
        "\n{:<20} │ {:>12} │ {:>12} │ {:>10} │ {:>20}",
        "Signal", "Expected", "Detected", "Lyapunov", "Branch Weights"
    );
    println!(
        "{:─<20}─┼─{:─^12}─┼─{:─^12}─┼─{:─^10}─┼─{:─^20}",
        "", "", "", "", ""
    );

    for (name, signal, expected_chaotic) in &signals {
        let (_acc, is_chaotic, lyap, weights) = test_advanced(signal, sample_rate as f32, name);

        let expected_str = if *expected_chaotic {
            "Chaotic"
        } else {
            "Regular"
        };
        let detected_str = if is_chaotic { "Chaotic ✓" } else { "Regular" };
        let status = if is_chaotic == *expected_chaotic {
            ""
        } else {
            " ⚠"
        };

        println!(
            "{:<20} │ {:>12} │ {:>10}{} │ {:>+9.4} │ [{:.2}, {:.2}, {:.2}]",
            name, expected_str, detected_str, status, lyap, weights[0], weights[1], weights[2]
        );
    }

    // Focus on chaotic signals improvement
    println!("\n{:=^90}", " CHAOTIC SIGNAL FOCUS ");

    let chaotic_signals = vec![
        ("Logistic r=3.8", generate_logistic(3.8, 1000)),
        ("Henon Map", generate_henon(1000)),
        ("Lorenz System", generate_lorenz(1000, 0.01)),
    ];

    println!(
        "\n{:<20} │ {:>10} │ {:>10} │ {:>10} │ {:>15}",
        "Chaotic Signal", "Baseline", "Enhanced", "Advanced", "Improvement"
    );
    println!(
        "{:─<20}─┼─{:─^10}─┼─{:─^10}─┼─{:─^10}─┼─{:─^15}",
        "", "", "", "", ""
    );

    for (name, signal) in &chaotic_signals {
        let baseline = test_baseline(signal, name);
        let enhanced = test_enhanced(signal, sample_rate as f32, name);
        let (advanced, _, _, _) = test_advanced(signal, sample_rate as f32, name);

        let improvement = (advanced - baseline) * 100.0;

        println!(
            "{:<20} │ {:>9.1}% │ {:>9.1}% │ {:>9.1}% │ {:>+14.1}%",
            name,
            baseline * 100.0,
            enhanced * 100.0,
            advanced * 100.0,
            improvement
        );
    }

    // Summary
    println!("\n{:=^90}", " SUMMARY ");
    println!(
        r#"
Implementation Progression:

  BASELINE (Original)     ENHANCED (+27%)          ADVANCED (+?%)
  ─────────────────────   ──────────────────────   ──────────────────────
  • Single time scale     • Multi-scale branches   • + Chaos detection
  • Binary threshold      • Amplitude encoding     • + Adaptive weights
  • Fixed learning        • Attention modulation   • + Memory horizon
                          • Fixed cycle detector   • + Lyapunov estimation
                                                   • + Delay embedding

Key Improvements in Advanced:

1. CHAOS DETECTION (Lyapunov Exponent)
   - Detects chaotic signals automatically
   - Routes to delay-embedding predictor
   - Expected +10-15% on chaotic signals

2. ADAPTIVE TIME CONSTANTS
   - Softmax temperature annealing
   - Branches specialize by signal type
   - Fixes "stuck at 33%" weight problem

3. MEMORY HORIZON
   - Multi-step prediction (1-5 steps)
   - Confidence decay for longer horizons
   - Catches longer-term patterns

4. AMPLITUDE-WEIGHTED LEARNING
   - High amplitude = faster learning
   - Focuses on significant transitions
"#
    );

    let improvement_overall = (avg_advanced - avg_baseline) * 100.0;
    let improvement_vs_enhanced = (avg_advanced - avg_enhanced) * 100.0;

    println!("\n  Results:");
    println!(
        "  ├─ Baseline → Enhanced:  {:>+6.1}%",
        (avg_enhanced - avg_baseline) * 100.0
    );
    println!(
        "  ├─ Enhanced → Advanced:  {:>+6.1}%",
        improvement_vs_enhanced
    );
    println!("  └─ Baseline → Advanced:  {:>+6.1}%", improvement_overall);

    println!(
        "\n╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                      ADVANCED COMPARISON COMPLETE                                    ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );
}