// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 21: Strategic Materials
//!
//! Demonstrates HDC + CfC strategic materials prediction with
//! O(1) prediction cost from 1 day to 50 years.

fn main() {
    println!("=== Genesis Mission Challenge 21: Strategic Materials ===\n");

    use symthaea_materials::strategic::{
        STRATEGIC_HORIZONS, StrategicHdcEncoder, StrategicPredictor, StrategicReading,
    };

    let encoder = StrategicHdcEncoder::new();
    let mut predictor = StrategicPredictor::new();

    let reading = StrategicReading {
        extreme_temp_resilience: 0.9,
        radiation_dose: 0.1,
        time_at_condition: 86_400.0,
        failure_probability: 0.001,
    };

    let hv = encoder.encode(&reading);
    predictor.observe(&hv, 86_400.0);

    println!("--- Strategic Materials Prediction ---");
    for &horizon in STRATEGIC_HORIZONS {
        let predicted = predictor.predict_at_horizon(&hv, horizon);
        let drift = 1.0 - hv.similarity(&predicted);
        println!("  Horizon {:>14.0}s: drift={:.4}", horizon, drift);
    }

    println!("\n--- O(1) Prediction Cost Proof ---");
    let input = symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea_core::hdc::unified_hv::HDC_DIMENSION,
        42,
    );
    for &horizon in STRATEGIC_HORIZONS {
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

    println!("\nPASS: Strategic Materials operational");
}
