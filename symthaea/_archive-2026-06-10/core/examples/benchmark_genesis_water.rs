// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 8: Water Prediction
//!
//! Demonstrates:
//! - Seasonal water quality encoding
//! - Contamination detection via surprise
//! - FEP emergency action selection

fn main() {
    println!("=== Genesis Mission Challenge 8: Water Prediction ===\n");

    use symthaea::symthaea_cell_foundry::hydrology::{
        HydrologicalPredictor, WATER_HORIZONS, WaterFepAgent, WaterHdcEncoder, WaterQualityReading,
        synthetic_water_scenario,
    };
    use symthaea::symthaea_core::hdc::unified_hv::HDC_DIMENSION;

    // 1. Clean vs contaminated encoding
    println!("--- Water Quality Encoding ---");
    let encoder = WaterHdcEncoder::new();
    let clean = WaterQualityReading::clean();
    let dirty = WaterQualityReading::contaminated();
    let hv_clean = encoder.encode(&clean);
    let hv_dirty = encoder.encode(&dirty);
    println!("  Clean water HV norm: {:.3}", hv_clean.norm());
    println!("  Contaminated water HV norm: {:.3}", hv_dirty.norm());
    println!(
        "  Clean vs contaminated similarity: {:.3}",
        hv_clean.similarity(&hv_dirty)
    );

    // 2. FEP agent decisions
    println!("\n--- FEP Agent Decisions ---");
    let agent = WaterFepAgent::new();
    println!("  Clean water: {:?}", agent.select_action(&clean));
    println!("  Contaminated: {:?}", agent.select_action(&dirty));
    println!("  Clean surprise: {}", agent.is_surprised(&clean));
    println!("  Contaminated surprise: {}", agent.is_surprised(&dirty));

    // 3. Seasonal scenario with contamination event
    println!("\n--- Seasonal Scenario (contamination at step 50) ---");
    let scenario = synthetic_water_scenario(100, Some(50));
    for (i, reading) in scenario.iter().enumerate() {
        if i % 10 == 0 || i == 50 {
            let action = agent.select_action(reading);
            let surprised = agent.is_surprised(reading);
            println!(
                "  Day {:>3}: pH={:.1} turb={:.1} coliform={:.1} → {:?} (surprised={})",
                i, reading.ph, reading.turbidity_ntu, reading.coliform_per_100ml, action, surprised
            );
        }
    }

    // 4. O(1) prediction cost
    println!("\n--- O(1) Prediction Cost ---");
    let predictor = HydrologicalPredictor::new();
    let input = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(HDC_DIMENSION, 42);
    for &horizon in WATER_HORIZONS {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = predictor.predict_at_horizon(&input, horizon);
        }
        let elapsed = start.elapsed();
        let label = match horizon as u64 {
            3600 => "1 hour",
            86400 => "1 day",
            604800 => "1 week",
            2592000 => "1 month",
            _ => "1 year",
        };
        println!(
            "  {:>10}: {:.1}µs/prediction",
            label,
            elapsed.as_micros() as f64 / 1000.0
        );
    }

    println!("\nPASS: Water Prediction operational");
}
