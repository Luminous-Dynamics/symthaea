// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Streaming analysis example - simulates real-time EEG processing

use sentinels::{AnalysisMode, analyze_consciousness};
use std::time::{Duration, Instant};

fn main() {
    println!("=== Real-Time Consciousness Monitoring ===");
    println!();

    let sample_rate = 256.0;
    let epoch_duration = 5.0; // Analyze every 5 seconds
    let total_duration = 30.0; // Run for 30 seconds

    let samples_per_epoch = (sample_rate * epoch_duration) as usize;
    let n_epochs = (total_duration / epoch_duration) as usize;

    println!("Sample Rate: {} Hz", sample_rate);
    println!("Epoch Duration: {} seconds", epoch_duration);
    println!("Total Epochs: {}", n_epochs);
    println!();
    println!(
        "{:<8} {:<15} {:<12} {:<12}",
        "Epoch", "State", "Conscious", "Wellbeing"
    );
    println!("{}", "-".repeat(50));

    for epoch in 0..n_epochs {
        let start = Instant::now();

        // Simulate EEG data collection
        // In real application, this would come from hardware
        let phase = epoch as f32 * 0.5; // Gradual change over time
        let data: Vec<f32> = (0..samples_per_epoch)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let pi2 = 2.0 * std::f32::consts::PI;

                // Simulate transition from alert to relaxed state
                let alpha_weight = 0.3 + 0.3 * (phase * 0.5).sin();
                let beta_weight = 0.3 - 0.2 * (phase * 0.5).sin();

                alpha_weight * (pi2 * 10.0 * t).sin()
                    + 0.2 * (pi2 * 6.0 * t).sin()
                    + beta_weight * (pi2 * 20.0 * t).sin()
                    + 0.05 * (pi2 * 2.0 * t).sin()
            })
            .collect();

        // Analyze
        match analyze_consciousness(&data, sample_rate, AnalysisMode::Auto) {
            Ok(poc) => {
                let elapsed = start.elapsed();
                println!(
                    "{:<8} {:<15} {:<12.1}% {:<12.1}%  ({:.2}ms)",
                    epoch + 1,
                    format!("{:?}", poc.state),
                    poc.consciousness_level * 100.0,
                    poc.wellbeing_score * 100.0,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                eprintln!("Epoch {}: Error - {}", epoch + 1, e);
            }
        }

        // Simulate real-time delay
        std::thread::sleep(Duration::from_millis(100));
    }

    println!();
    println!("Streaming complete.");
}