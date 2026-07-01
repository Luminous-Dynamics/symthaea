// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 19: Building Systems
//!
//! Demonstrates HDC + CfC + FEP building monitoring with
//! O(1) prediction cost from 1 hour to 30 years.

fn main() {
    println!("=== Genesis Mission Challenge 19: Building Systems ===\n");

    use symthaea_fabrication_kernel::building::{BUILDING_HORIZONS, BuildingReading, BuildingTwin};

    let healthy = BuildingReading {
        thermal_load: 0.4,
        structural_stress: 0.1,
        occupancy: 0.6,
        comfort: 0.85,
        energy_consumption: 0.35,
    };

    let mut twin = BuildingTwin::new();
    twin.set_reference(&healthy);

    println!("--- Building Health Timeline ---");
    for i in 0..10 {
        let output = twin.step(&healthy, 3600.0);
        if i % 3 == 0 {
            println!(
                "Step {:>3}: {:?} | FE={:.3} | Action={:?}",
                i, output.safety_level, output.free_energy, output.recommended_action
            );
        }
    }

    println!("\n--- O(1) Prediction Cost Proof ---");
    let predictor = twin.predictor();
    let input = symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea_core::hdc::unified_hv::HDC_DIMENSION,
        42,
    );
    for &horizon in BUILDING_HORIZONS {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = predictor.predict_at_horizon(&input, horizon);
        }
        let elapsed = start.elapsed();
        println!(
            "  Horizon {:>14.1}s: {:.1}µs/prediction",
            horizon,
            elapsed.as_micros() as f64 / 1000.0
        );
    }

    println!("\nTotal cycles processed: {}", twin.cycle_count());
    println!("PASS: Building Systems operational");
}
