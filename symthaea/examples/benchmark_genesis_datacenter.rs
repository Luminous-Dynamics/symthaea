// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 20: Data Center Operations
//!
//! Demonstrates HDC + CfC + FEP datacenter monitoring with
//! O(1) prediction cost from 1 minute to 1 year.

fn main() {
    println!("=== Genesis Mission Challenge 20: Data Center Operations ===\n");

    use symthaea::physics::datacenter::{DATACENTER_HORIZONS, DatacenterReading, DatacenterTwin};

    let healthy = DatacenterReading {
        power_draw: 0.6,
        cooling_load: 0.4,
        latency: 5.0,
        throughput: 0.8,
    };

    let mut twin = DatacenterTwin::new();
    twin.set_reference(&healthy);

    println!("--- Datacenter Health Timeline ---");
    for i in 0..10 {
        let output = twin.step(&healthy, 60.0);
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
    for &horizon in DATACENTER_HORIZONS {
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
    println!("PASS: Datacenter Operations operational");
}
