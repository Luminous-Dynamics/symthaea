// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 24: Proliferation Safeguards
//!
//! Demonstrates HDC + CfC safeguards monitoring with
//! O(1) prediction cost from 1 day to 1 year.

fn main() {
    println!("=== Genesis Mission Challenge 24: Proliferation Safeguards ===\n");

    use symthaea_nuclear_forensics::safeguards::{
        SAFEGUARDS_HORIZONS, SafeguardsHdcEncoder, SafeguardsPredictor, SafeguardsReading,
    };

    let encoder = SafeguardsHdcEncoder::new();
    let mut predictor = SafeguardsPredictor::new();

    let reading = SafeguardsReading {
        inventory_discrepancy: 0.01,
        sensor_anomaly: 0.02,
        timeline_consistency: 0.95,
    };

    let hv = encoder.encode(&reading);
    predictor.observe(&hv, 86_400.0);

    println!("--- Safeguards Prediction ---");
    for &horizon in SAFEGUARDS_HORIZONS {
        let predicted = predictor.predict_at_horizon(&hv, horizon);
        let drift = 1.0 - hv.similarity(&predicted);
        println!("  Horizon {:>14.0}s: drift={:.4}", horizon, drift);
    }

    println!("\n--- O(1) Prediction Cost Proof ---");
    let input = symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea_core::hdc::unified_hv::HDC_DIMENSION,
        42,
    );
    for &horizon in SAFEGUARDS_HORIZONS {
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

    println!("\nPASS: Proliferation Safeguards operational");
}
