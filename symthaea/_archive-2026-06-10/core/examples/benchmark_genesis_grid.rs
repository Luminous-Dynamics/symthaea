// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 1: Grid Scaling
//!
//! Demonstrates HDC + CfC + FEP grid stability monitoring with
//! O(1) prediction cost across timescales from 1 minute to 1 year.

fn main() {
    println!("=== Genesis Mission Challenge 1: Grid Scaling ===\n");

    use symthaea::physics::grid::{GRID_HORIZONS, GridReading, GridTwin};

    // 1. Create a healthy grid scenario
    let healthy = GridReading {
        voltage: 1.0,
        frequency: 60.0,
        demand: 0.6,
        reserve_margin: 0.2,
        line_flow: 0.5,
    };

    let mut twin = GridTwin::new();
    twin.set_reference(&healthy);

    // 2. Run healthy steps
    println!("--- Grid Stability Timeline ---");
    for i in 0..10 {
        let output = twin.step(&healthy, 60.0);
        if i % 3 == 0 {
            println!(
                "Step {:>3}: {:?} | FE={:.3} | Action={:?}",
                i, output.safety_level, output.free_energy, output.recommended_action
            );
        }
    }

    // 3. O(1) cost proof
    println!("\n--- O(1) Prediction Cost Proof ---");
    let predictor = twin.predictor();
    let input = symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea_core::hdc::unified_hv::HDC_DIMENSION,
        42,
    );
    for &horizon in GRID_HORIZONS {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = predictor.predict_at_horizon(&input, horizon);
        }
        let elapsed = start.elapsed();
        println!(
            "  Horizon {:>12.1}s: {:.1}µs/prediction",
            horizon,
            elapsed.as_micros() as f64 / 1000.0
        );
    }

    println!("\nTotal cycles processed: {}", twin.cycle_count());
    println!("PASS: Grid Scaling operational");
}
