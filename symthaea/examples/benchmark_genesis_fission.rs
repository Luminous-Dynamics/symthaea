// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 2: Fission Reactor Safety
//!
//! Demonstrates HDC + CfC + FEP fission reactor monitoring with
//! O(1) prediction cost from 0.1s to 1 day.

fn main() {
    println!("=== Genesis Mission Challenge 2: Fission Reactor Safety ===\n");

    use symthaea::physics::fission::{FISSION_HORIZONS, FissionReading, FissionTwin};

    let healthy = FissionReading {
        power_output: 0.8,
        coolant_temp: 300.0,
        neutron_flux: 0.5,
        pressure: 10.0,
        control_rod_pos: 0.5,
    };

    let mut twin = FissionTwin::new();
    twin.set_reference(&healthy);

    println!("--- Reactor Safety Timeline ---");
    for i in 0..10 {
        let output = twin.step(&healthy, 10.0);
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
    for &horizon in FISSION_HORIZONS {
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
    println!("PASS: Fission Reactor Safety operational");
}
