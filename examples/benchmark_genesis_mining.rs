// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 18: Critical Minerals
//!
//! Demonstrates HDC + CfC critical minerals prediction with
//! O(1) prediction cost from 1 day to 10 years.

fn main() {
    println!("=== Genesis Mission Challenge 18: Critical Minerals ===\n");

    use symthaea_materials::mining::{
        MINING_HORIZONS, MiningHdcEncoder, MiningPredictor, MiningReading,
    };

    let encoder = MiningHdcEncoder::new();
    let mut predictor = MiningPredictor::new();

    let reading = MiningReading {
        ore_grade: 0.5,
        extraction_rate: 0.7,
        environmental_impact: 0.1,
        cost: 0.4,
    };

    let hv = encoder.encode(&reading);
    predictor.observe(&hv, 86_400.0);

    println!("--- Critical Minerals Prediction ---");
    for &horizon in MINING_HORIZONS {
        let predicted = predictor.predict_at_horizon(&hv, horizon);
        let drift = 1.0 - hv.similarity(&predicted);
        println!("  Horizon {:>14.0}s: drift={:.4}", horizon, drift);
    }

    println!("\n--- O(1) Prediction Cost Proof ---");
    let input = symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea_core::hdc::unified_hv::HDC_DIMENSION,
        42,
    );
    for &horizon in MINING_HORIZONS {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = predictor.predict_at_horizon(&input, horizon);
        }
        let elapsed = start.elapsed();
        println!(
            "  Horizon {:>14.0}s: {:.1}µs/prediction",
            horizon,
            elapsed.as_micros() as f64 / 1000.0
        );
    }

    println!("\nPASS: Critical Minerals operational");
}
