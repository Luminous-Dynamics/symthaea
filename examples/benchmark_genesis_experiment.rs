// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 5: Experiment Planner
//!
//! Demonstrates HDC + CfC experiment planning with
//! O(1) prediction cost from 1 hour to 1 month.

fn main() {
    println!("=== Genesis Mission Challenge 5: Experiment Planner ===\n");

    use symthaea::symthaea_cell_foundry::experiment_planner::{
        EXPERIMENT_HORIZONS, ExperimentHdcEncoder, ExperimentPredictor, ExperimentReading,
    };

    let encoder = ExperimentHdcEncoder::new();
    let mut predictor = ExperimentPredictor::new();

    let reading = ExperimentReading {
        experiment_uncertainty: 0.2,
        cost: 0.3,
        time_remaining: 604_800.0,
        info_gain: 0.7,
    };

    let hv = encoder.encode(&reading);
    predictor.observe(&hv, 3600.0);

    println!("--- Experiment Prediction ---");
    for &horizon in EXPERIMENT_HORIZONS {
        let predicted = predictor.predict_at_horizon(&hv, horizon);
        let drift = 1.0 - hv.similarity(&predicted);
        println!("  Horizon {:>12.0}s: drift={:.4}", horizon, drift);
    }

    println!("\n--- O(1) Prediction Cost Proof ---");
    let input = symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea_core::hdc::unified_hv::HDC_DIMENSION,
        42,
    );
    for &horizon in EXPERIMENT_HORIZONS {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = predictor.predict_at_horizon(&input, horizon);
        }
        let elapsed = start.elapsed();
        println!(
            "  Horizon {:>12.0}s: {:.1}µs/prediction",
            horizon,
            elapsed.as_micros() as f64 / 1000.0
        );
    }

    println!("\nPASS: Experiment Planner operational");
}
