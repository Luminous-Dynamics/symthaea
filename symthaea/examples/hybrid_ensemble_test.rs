// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test Hybrid Ensemble Predictor on chaotic signals
//!
//! Validates that the ensemble correctly routes:
//! - Continuous chaos (Lorenz) → ESN (93.8%)
//! - Discrete chaos (Logistic/Henon) → LTC (75-80%)
//!
//! Expected: Ensemble should match or exceed individual method performance

use symthaea::hdc::reservoir::{HybridEnsemblePredictor, SignalType};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       HYBRID ENSEMBLE PREDICTOR - CHAOTIC SIGNAL TEST        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Routes signals to optimal method based on characteristics   ║");
    println!("║  - ESN for continuous chaos (Lorenz: 93.8%)                  ║");
    println!("║  - LTC for discrete chaos (Logistic/Henon: ~75%)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test 1: Logistic Map r=3.8 (Discrete Chaotic)
    println!("=== LOGISTIC MAP r=3.8 (DISCRETE CHAOTIC) ===\n");
    let (log_acc, log_type) = test_logistic_map();
    println!("  Detected type:     {:?}", log_type);
    println!("  Expected type:     DiscreteChaos");
    let log_correct = matches!(log_type, SignalType::DiscreteChaos);
    println!(
        "  Routing correct:   {}\n",
        if log_correct { "✅ YES" } else { "⚠️ NO" }
    );

    // Test 2: Henon Map (Discrete Chaotic)
    println!("=== HENON MAP (DISCRETE CHAOTIC) ===\n");
    let (hen_acc, hen_type) = test_henon_map();
    println!("  Detected type:     {:?}", hen_type);
    println!("  Expected type:     DiscreteChaos");
    let hen_correct = matches!(hen_type, SignalType::DiscreteChaos);
    println!(
        "  Routing correct:   {}\n",
        if hen_correct { "✅ YES" } else { "⚠️ NO" }
    );

    // Test 3: Lorenz System (Continuous Chaotic)
    println!("=== LORENZ SYSTEM (CONTINUOUS CHAOTIC) ===\n");
    let (lor_acc, lor_type) = test_lorenz();
    println!("  Detected type:     {:?}", lor_type);
    println!("  Expected type:     ContinuousChaos");
    let lor_correct = matches!(lor_type, SignalType::ContinuousChaos);
    println!(
        "  Routing correct:   {}\n",
        if lor_correct { "✅ YES" } else { "⚠️ NO" }
    );

    // Summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      SUMMARY                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Signal           Accuracy   Routing   Target                ║");
    println!("║  ─────────────────────────────────────────────────────────── ║");
    println!(
        "║  Logistic r=3.8   {:>5.1}%     {:>6}    75%+                  ║",
        log_acc,
        if log_correct { "✅" } else { "⚠️" }
    );
    println!(
        "║  Henon Map        {:>5.1}%     {:>6}    75%+                  ║",
        hen_acc,
        if hen_correct { "✅" } else { "⚠️" }
    );
    println!(
        "║  Lorenz System    {:>5.1}%     {:>6}    90%+                  ║",
        lor_acc,
        if lor_correct { "✅" } else { "⚠️" }
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Overall assessment
    let all_routed = log_correct && hen_correct && lor_correct;
    let avg_acc = (log_acc + hen_acc + lor_acc) / 3.0;
    println!(
        "\n  Signal Type Detection: {}",
        if all_routed {
            "✅ ALL CORRECT"
        } else {
            "⚠️ NEEDS TUNING"
        }
    );
    println!("  Average Accuracy:      {:.1}%", avg_acc);
    println!(
        "  Overall Assessment:    {}",
        if all_routed && avg_acc > 70.0 {
            "✅ ENSEMBLE WORKING"
        } else if avg_acc > 60.0 {
            "🔄 PARTIAL SUCCESS"
        } else {
            "⚠️ NEEDS IMPROVEMENT"
        }
    );
}

fn test_logistic_map() -> (f64, SignalType) {
    let r = 3.8;
    let mut x = 0.1;

    // Generate data
    let mut data = Vec::new();
    for _ in 0..3000 {
        data.push(x);
        x = r * x * (1.0 - x);
    }

    // Create hybrid ensemble
    let mut ensemble = HybridEnsemblePredictor::new(42);

    // Training phase (first 1500 samples)
    for sample in data.iter().take(1500) {
        ensemble.observe(*sample);
    }

    // Force signal type detection
    for _ in 0..200 {
        ensemble.observe(data[1499]);
    }

    let signal_type = ensemble.get_signal_type();

    // Reset for clean testing
    let mut ensemble = HybridEnsemblePredictor::new(42);
    for sample in data.iter().take(1500) {
        ensemble.observe(*sample);
    }

    // Test phase
    let mut correct = 0;
    let mut total = 0;
    let threshold = 0.5;

    for i in 1500..2499 {
        let pred_value = ensemble.predict();
        let pred_binary = pred_value > threshold;
        let actual_binary = data[i + 1] > threshold;

        if pred_binary == actual_binary {
            correct += 1;
        }
        total += 1;

        ensemble.observe(data[i]);
    }

    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("  Training samples:  1500");
    println!("  Test samples:      {}", total);
    println!("  Correct:           {}", correct);
    println!("  Accuracy:          {:.1}%", accuracy);
    println!("  (Random baseline:  50.0%)");
    println!("  Diagnostics:       {}", ensemble.diagnostics());

    if accuracy > 60.0 {
        println!("  Result:            ✅ BETTER THAN RANDOM");
    } else if accuracy > 52.0 {
        println!("  Result:            🔄 MARGINALLY BETTER");
    } else {
        println!("  Result:            ⚠️ Near random");
    }

    (accuracy, signal_type)
}

fn test_henon_map() -> (f64, SignalType) {
    let a = 1.4;
    let b = 0.3;
    let mut x = 0.1;
    let mut y = 0.1;

    // Generate data
    let mut data = Vec::new();
    for _ in 0..3000 {
        data.push(x);
        let new_x = 1.0 - a * x * x + y;
        let new_y = b * x;
        x = new_x;
        y = new_y;
    }

    // Normalize
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    let data: Vec<f64> = data.iter().map(|v| (v - min_val) / range).collect();

    // Create hybrid ensemble
    let mut ensemble = HybridEnsemblePredictor::new(123);

    // Training phase
    for sample in data.iter().take(1500) {
        ensemble.observe(*sample);
    }

    let signal_type = ensemble.get_signal_type();

    // Test phase
    let mut correct = 0;
    let mut total = 0;

    for i in 1500..2499 {
        let pred_binary = ensemble.predict() > 0.5;
        let actual_binary = data[i + 1] > 0.5;

        if pred_binary == actual_binary {
            correct += 1;
        }
        total += 1;
        ensemble.observe(data[i]);
    }

    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("  Training samples:  1500");
    println!("  Test samples:      {}", total);
    println!("  Correct:           {}", correct);
    println!("  Accuracy:          {:.1}%", accuracy);
    println!("  (Random baseline:  50.0%)");
    println!("  Diagnostics:       {}", ensemble.diagnostics());

    if accuracy > 60.0 {
        println!("  Result:            ✅ BETTER THAN RANDOM");
    } else if accuracy > 52.0 {
        println!("  Result:            🔄 MARGINALLY BETTER");
    } else {
        println!("  Result:            ⚠️ Near random");
    }

    (accuracy, signal_type)
}

fn test_lorenz() -> (f64, SignalType) {
    // Lorenz system
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;
    let dt = 0.01;

    let mut x = 1.0;
    let mut y = 1.0;
    let mut z = 1.0;

    // Generate data
    let mut data = Vec::new();
    for _ in 0..20000 {
        data.push(x);
        let dx = sigma * (y - x);
        let dy = x * (rho - z) - y;
        let dz = x * y - beta * z;
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;
    }

    // Subsample and normalize
    let data: Vec<f64> = data.iter().step_by(10).cloned().collect();
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    let data: Vec<f64> = data.iter().map(|v| (v - min_val) / range).collect();

    // Create hybrid ensemble
    let mut ensemble = HybridEnsemblePredictor::new(456);

    // Training phase
    let train_size = (data.len() * 3 / 4).min(1500);
    for sample in data.iter().take(train_size) {
        ensemble.observe(*sample);
    }

    let signal_type = ensemble.get_signal_type();

    // Test phase
    let mut correct = 0;
    let mut total = 0;

    for i in train_size..(data.len() - 1) {
        let pred_binary = ensemble.predict() > 0.5;
        let actual_binary = data[i + 1] > 0.5;

        if pred_binary == actual_binary {
            correct += 1;
        }
        total += 1;
        ensemble.observe(data[i]);
    }

    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("  Training samples:  {}", train_size);
    println!("  Test samples:      {}", total);
    println!("  Correct:           {}", correct);
    println!("  Accuracy:          {:.1}%", accuracy);
    println!("  (Random baseline:  50.0%)");
    println!("  Diagnostics:       {}", ensemble.diagnostics());

    if accuracy > 80.0 {
        println!("  Result:            ✅ EXCELLENT");
    } else if accuracy > 60.0 {
        println!("  Result:            ✅ BETTER THAN RANDOM");
    } else {
        println!("  Result:            ⚠️ Near random");
    }

    (accuracy, signal_type)
}