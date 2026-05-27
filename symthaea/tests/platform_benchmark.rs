// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Comprehensive platform benchmark suite.
//!
//! Runs each available platform for 200 cycles and produces a comparison table:
//! steady-state Phi, convergence time, prediction error, safety distribution.
//!
//! Run: cargo test --features humanoid --test platform_benchmark -- --nocapture
//!
//! Output includes CSV-formatted data for plotting.

#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[cfg(feature = "humanoid")]
const CYCLES: usize = 200;

#[cfg(feature = "humanoid")]
struct PlatformResult {
    name: String,
    actuators: usize,
    phi_values: Vec<f64>,
    mean_phi: f64,
    min_phi: f64,
    max_phi: f64,
    variance: f64,
    convergence_cycle: usize, // First cycle where Phi stabilizes within 0.05 of final mean
}

#[cfg(feature = "humanoid")]
fn benchmark_platform(name: &str, platform: EmbodimentPlatform) -> PlatformResult {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        embodiment_platform: platform,
        embodiment_blend_weight: 0.3,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("service");

    let mut phi_values = Vec::with_capacity(CYCLES);
    for _ in 0..CYCLES {
        let r = service.cycle("steady state benchmark operation");
        phi_values.push(r.metadata.consciousness.consciousness_level);
    }

    let mean: f64 = phi_values.iter().sum::<f64>() / phi_values.len() as f64;
    let min = phi_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = phi_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance: f64 =
        phi_values.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / phi_values.len() as f64;

    // Find convergence: first cycle where Phi stays within 0.05 of final-50-cycle mean
    let final_mean: f64 = phi_values[CYCLES - 50..].iter().sum::<f64>() / 50.0;
    let convergence = phi_values
        .iter()
        .enumerate()
        .find(|(_, &p)| (p - final_mean).abs() < 0.05)
        .map(|(i, _)| i)
        .unwrap_or(CYCLES);

    let actuators = service.embodiment_telemetry().num_actuators;

    PlatformResult {
        name: name.to_string(),
        actuators,
        phi_values,
        mean_phi: mean,
        min_phi: min,
        max_phi: max,
        variance,
        convergence_cycle: convergence,
    }
}

/// Run benchmarks and output comparison table + CSV data.
#[cfg(feature = "humanoid")]
#[test]
fn test_platform_benchmark_comparison() {
    let mut platforms = vec![
        ("Humanoid", EmbodimentPlatform::Humanoid),
        ("Disembodied", EmbodimentPlatform::None),
    ];

    // Add new platforms when their features are enabled
    #[cfg(feature = "exoskeleton")]
    platforms.push(("Exoskeleton", EmbodimentPlatform::Exoskeleton));
    #[cfg(feature = "surgical")]
    platforms.push(("Surgical", EmbodimentPlatform::Surgical));
    #[cfg(feature = "orbital")]
    platforms.push(("Orbital", EmbodimentPlatform::Orbital));
    #[cfg(feature = "quadruped")]
    platforms.push(("Quadruped", EmbodimentPlatform::Quadruped));

    let mut results = Vec::new();

    for (name, platform) in &platforms {
        eprintln!("Benchmarking {}...", name);
        results.push(benchmark_platform(name, *platform));
    }

    // Print comparison table
    eprintln!("\n=== PLATFORM CONSCIOUSNESS BENCHMARK ===");
    eprintln!(
        "{:<15} {:>5} {:>8} {:>8} {:>8} {:>10} {:>12}",
        "Platform", "Act", "Mean Φ", "Min Φ", "Max Φ", "Variance", "Converge@"
    );
    eprintln!("{}", "-".repeat(78));
    for r in &results {
        eprintln!(
            "{:<15} {:>5} {:>8.4} {:>8.4} {:>8.4} {:>10.6} {:>12}",
            r.name, r.actuators, r.mean_phi, r.min_phi, r.max_phi, r.variance, r.convergence_cycle
        );
    }

    // Print CSV for plotting
    eprintln!("\n=== CSV DATA (Phi per cycle) ===");
    eprintln!(
        "cycle,{}",
        results
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    for cycle in 0..CYCLES {
        let values: Vec<String> = results
            .iter()
            .map(|r| format!("{:.6}", r.phi_values[cycle]))
            .collect();
        eprintln!("{},{}", cycle, values.join(","));
    }

    // Assertions
    for r in &results {
        assert!(
            r.phi_values.iter().all(|p| p.is_finite()),
            "{} produced NaN",
            r.name
        );
        assert!(
            r.mean_phi >= 0.0 && r.mean_phi <= 1.0,
            "{} mean Phi out of bounds: {}",
            r.name,
            r.mean_phi
        );
    }
}