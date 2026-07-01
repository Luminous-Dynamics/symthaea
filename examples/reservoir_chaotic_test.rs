// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test Echo State Network on chaotic signals
//!
//! Compares ESN performance vs baseline on:
//! - Logistic map r=3.8 (chaotic)
//! - Henon map (chaotic)
//!
//! Expected: ESN should achieve 60-80% vs 50% random baseline

use symthaea::hdc::reservoir::EchoStateNetwork;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         RESERVOIR COMPUTING - CHAOTIC SIGNAL TEST            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test 1: Logistic Map r=3.8 (Chaotic)
    println!("=== LOGISTIC MAP r=3.8 (CHAOTIC) ===\n");
    test_logistic_map();

    // Test 2: Henon Map (Chaotic)
    println!("\n=== HENON MAP (CHAOTIC) ===\n");
    test_henon_map();

    // Test 3: Lorenz System (Chaotic)
    println!("\n=== LORENZ SYSTEM (CHAOTIC) ===\n");
    test_lorenz();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    TEST COMPLETE                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn test_logistic_map() {
    let r = 3.8;
    let mut x = 0.1;

    // Generate more data for discrete maps
    let mut data = Vec::new();
    for _ in 0..5000 {
        data.push(x);
        x = r * x * (1.0 - x);
    }

    // Use ESN optimized for discrete chaotic maps:
    // - Small reservoir (50 neurons)
    // - High spectral radius (0.99) for edge of chaos
    // - Low leaking rate (0.1) for discrete dynamics
    // - Short warmup (50 steps)
    let mut esn = EchoStateNetwork::for_discrete_chaos(42);

    // Training phase (first 1500 samples)
    for i in 0..1500 {
        let target = if i + 1 < data.len() {
            Some(data[i + 1])
        } else {
            None
        };
        esn.observe(data[i], target);
    }
    esn.train(0.001);

    // Reset and warm up on training data
    esn.reset_state();
    for sample in data.iter().take(1500) {
        esn.observe(*sample, None);
    }

    // Test phase (last 500 samples)
    let mut correct = 0;
    let mut total = 0;
    let threshold = 0.5; // Logistic map is in [0, 1]

    for i in 1500..1999 {
        let pred_value = esn.predict();
        let pred_binary = pred_value > threshold;
        let actual_binary = data[i + 1] > threshold;

        if pred_binary == actual_binary {
            correct += 1;
        }
        total += 1;

        // Update ESN with actual value
        esn.observe(data[i], None);
    }

    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("  Training samples:  1500");
    println!("  Test samples:      {}", total);
    println!("  Correct:           {}", correct);
    println!("  Accuracy:          {:.1}%", accuracy);
    println!("  (Random baseline:  50.0%)");

    if accuracy > 55.0 {
        println!("  Result:            ✅ BETTER THAN RANDOM");
    } else {
        println!("  Result:            ⚠️  Near random");
    }
}

fn test_henon_map() {
    // Henon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n
    let a = 1.4;
    let b = 0.3;
    let mut x = 0.1;
    let mut y = 0.1;

    // Generate data (use x coordinate)
    let mut data = Vec::new();
    for _ in 0..2000 {
        data.push(x);
        let new_x = 1.0 - a * x * x + y;
        let new_y = b * x;
        x = new_x;
        y = new_y;
    }

    // Normalize to [0, 1] for binary prediction
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    let data: Vec<f64> = data.iter().map(|v| (v - min_val) / range).collect();

    // Use ESN optimized for discrete chaotic maps
    let mut esn = EchoStateNetwork::for_discrete_chaos(123);

    for i in 0..1500 {
        let target = if i + 1 < data.len() {
            Some(data[i + 1])
        } else {
            None
        };
        esn.observe(data[i], target);
    }
    esn.train(0.001);

    esn.reset_state();
    for sample in data.iter().take(1500) {
        esn.observe(*sample, None);
    }

    let mut correct = 0;
    let mut total = 0;

    for i in 1500..1999 {
        let pred_binary = esn.predict() > 0.5;
        let actual_binary = data[i + 1] > 0.5;

        if pred_binary == actual_binary {
            correct += 1;
        }
        total += 1;
        esn.observe(data[i], None);
    }

    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("  Training samples:  1500");
    println!("  Test samples:      {}", total);
    println!("  Correct:           {}", correct);
    println!("  Accuracy:          {:.1}%", accuracy);
    println!("  (Random baseline:  50.0%)");

    if accuracy > 55.0 {
        println!("  Result:            ✅ BETTER THAN RANDOM");
    } else {
        println!("  Result:            ⚠️  Near random");
    }
}

fn test_lorenz() {
    // Lorenz system (simplified Euler integration)
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;
    let dt = 0.01;

    let mut x = 1.0;
    let mut y = 1.0;
    let mut z = 1.0;

    // Generate data (use x coordinate)
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

    // Subsample to reduce autocorrelation
    let data: Vec<f64> = data.iter().step_by(10).cloned().collect();

    // Normalize
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    let data: Vec<f64> = data.iter().map(|v| (v - min_val) / range).collect();

    // Train ESN
    let mut esn = EchoStateNetwork::new(150, 0.95, 0.85, 456);

    let train_size = (data.len() * 3 / 4).min(1500);
    for i in 0..train_size {
        let target = if i + 1 < data.len() {
            Some(data[i + 1])
        } else {
            None
        };
        esn.observe(data[i], target);
    }
    esn.train(0.0001);

    esn.reset_state();
    for sample in data.iter().take(train_size) {
        esn.observe(*sample, None);
    }

    let mut correct = 0;
    let mut total = 0;

    for i in train_size..(data.len() - 1) {
        let pred_binary = esn.predict() > 0.5;
        let actual_binary = data[i + 1] > 0.5;

        if pred_binary == actual_binary {
            correct += 1;
        }
        total += 1;
        esn.observe(data[i], None);
    }

    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("  Training samples:  {}", train_size);
    println!("  Test samples:      {}", total);
    println!("  Correct:           {}", correct);
    println!("  Accuracy:          {:.1}%", accuracy);
    println!("  (Random baseline:  50.0%)");

    if accuracy > 55.0 {
        println!("  Result:            ✅ BETTER THAN RANDOM");
    } else {
        println!("  Result:            ⚠️  Near random");
    }
}