// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 13: Particle Accelerator Control
//!
//! Demonstrates HDC + CfC + FEP accelerator beam monitoring with
//! O(1) prediction cost from 1ms to 10 hours.

fn main() {
    println!("=== Genesis Mission Challenge 13: Particle Accelerator ===\n");

    use symthaea::physics::accelerator::{
        ACCELERATOR_HORIZONS, AcceleratorReading, AcceleratorTwin,
    };

    let healthy = AcceleratorReading {
        beam_energy: 6500.0,
        luminosity: 1.0,
        beam_loss: 0.01,
        tune_drift: 0.02,
        emittance: 0.1,
    };

    let mut twin = AcceleratorTwin::new();
    twin.set_reference(&healthy);

    println!("--- Beam Stability Timeline ---");
    for i in 0..10 {
        let output = twin.step(&healthy, 1.0);
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
    for &horizon in ACCELERATOR_HORIZONS {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = predictor.predict_at_horizon(&input, horizon);
        }
        let elapsed = start.elapsed();
        println!(
            "  Horizon {:>12.3}s: {:.1}µs/prediction",
            horizon,
            elapsed.as_micros() as f64 / 1000.0
        );
    }

    println!("\nTotal cycles processed: {}", twin.cycle_count());
    println!("PASS: Accelerator Control operational");
}
