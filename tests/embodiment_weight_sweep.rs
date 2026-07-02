// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Embodiment blend weight sweep — maps Phi vs weight to find optimal point.
//! Run: cargo test --features humanoid --test embodiment_weight_sweep -- --nocapture

#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[cfg(feature = "humanoid")]
const CYCLES: usize = 150;

#[cfg(feature = "humanoid")]
#[test]
fn test_embodiment_weight_sweep() {
    let weights: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
    let mut results = Vec::new();

    eprintln!("\n=== EMBODIMENT WEIGHT SWEEP ===");
    eprintln!(
        "{:<8} {:>8} {:>8} {:>8} {:>10}",
        "Weight", "Mean Φ", "Min Φ", "Max Φ", "Variance"
    );
    eprintln!("{}", "-".repeat(52));

    for &w in &weights {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            embodiment_platform: EmbodimentPlatform::Humanoid,
            embodiment_blend_weight: w as f32,
            embodiment_step_interval: 1,
            async_training: false,
            learning_threshold: 0.0,
            ..Default::default()
        })
        .expect("service");

        let mut phi_vals = Vec::with_capacity(CYCLES);
        for _ in 0..CYCLES {
            let r = service.cycle("operating with embodiment feedback");
            phi_vals.push(r.metadata.consciousness.consciousness_level);
        }

        // Use last 50 cycles for steady-state stats
        let steady = &phi_vals[CYCLES - 50..];
        let mean: f64 = steady.iter().sum::<f64>() / steady.len() as f64;
        let min = steady.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = steady.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let var: f64 = steady.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / steady.len() as f64;

        eprintln!(
            "{:<8.1} {:>8.4} {:>8.4} {:>8.4} {:>10.6}",
            w, mean, min, max, var
        );
        results.push((w, mean, min, max, var));
    }

    // CSV output for plotting
    eprintln!("\n=== CSV ===");
    eprintln!("weight,mean_phi,min_phi,max_phi,variance");
    for (w, mean, min, max, var) in &results {
        eprintln!("{:.1},{:.6},{:.6},{:.6},{:.6}", w, mean, min, max, var);
    }

    // Find optimal weight (highest mean Phi)
    let best = results.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    eprintln!("\nOPTIMAL: weight={:.1} → mean Phi={:.4}", best.0, best.1);

    // All values must be finite
    for (w, mean, _, _, _) in &results {
        assert!(mean.is_finite(), "Weight {w} produced NaN");
    }
}
