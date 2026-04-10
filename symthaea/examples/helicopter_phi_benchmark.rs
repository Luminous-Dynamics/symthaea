// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helicopter phi-correlation benchmark: consciousness predicts flight performance.
//!
//! Runs hover benchmarks across 11 consciousness levels and exports CSV data
//! for the paper claim: "Phi predicts perturbation recovery (r=0.87 on humanoid)".
//!
//! Usage: cargo run --features helicopter --example helicopter_phi_benchmark

fn main() {
    use symthaea_helicopter::benchmarks;
    use symthaea_helicopter::types::HelicopterConfig;

    let config = HelicopterConfig {
        steps_per_episode: 900, // 3 seconds at 300Hz
        ..Default::default()
    };

    println!(
        "phi,mean_alt_error,control_effort,angular_speed,hover_fraction,crashed,steps_survived"
    );

    let results = benchmarks::phi_correlation_sweep(&config, 20.0, 900);

    for (phi, result) in &results {
        println!(
            "{:.2},{:.4},{:.4},{:.4},{:.4},{},{}",
            phi,
            result.mean_altitude_error,
            result.mean_control_effort,
            result.mean_angular_speed,
            result.hover_fraction,
            result.crashed,
            result.steps_survived,
        );
    }

    // Compute Pearson correlation: phi vs hover_fraction
    let n = results.len() as f64;
    let mean_phi: f64 = results.iter().map(|(p, _)| p).sum::<f64>() / n;
    let mean_hover: f64 = results.iter().map(|(_, r)| r.hover_fraction).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_phi = 0.0;
    let mut var_hover = 0.0;
    for (phi, result) in &results {
        let dp = phi - mean_phi;
        let dh = result.hover_fraction - mean_hover;
        cov += dp * dh;
        var_phi += dp * dp;
        var_hover += dh * dh;
    }

    let r = if var_phi > 0.0 && var_hover > 0.0 {
        cov / (var_phi.sqrt() * var_hover.sqrt())
    } else {
        0.0
    };

    eprintln!("\n=== Phi-Hover Correlation ===");
    eprintln!("Pearson r = {r:.4}");
    eprintln!("N = {}", results.len());
    eprintln!(
        "Interpretation: {}",
        if r > 0.7 {
            "Strong positive — consciousness predicts flight stability"
        } else if r > 0.3 {
            "Moderate positive"
        } else {
            "Weak or no correlation"
        }
    );
}
