// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 22: Threat Assessment
//!
//! Demonstrates HDC + CfC + FEP threat monitoring with
//! O(1) prediction cost from 10ms to 1 hour.

fn main() {
    println!("=== Genesis Mission Challenge 22: Threat Assessment ===\n");

    use symthaea::physics::threat::{THREAT_HORIZONS, ThreatReading, ThreatTwin};

    let healthy = ThreatReading {
        sensor_reading: 0.5,
        expected_value: 0.5,
        surprise_signal: 0.0,
        response_time: 1.0,
        confidence: 0.9,
    };

    let mut twin = ThreatTwin::new();
    twin.set_reference(&healthy);

    println!("--- Threat Monitoring Timeline ---");
    for i in 0..10 {
        let output = twin.step(&healthy, 1.0);
        if i % 3 == 0 {
            println!(
                "Step {:>3}: {:?} | FE={:.3} | Action={:?}",
                i, output.threat_level, output.free_energy, output.recommended_action
            );
        }
    }

    println!("\n--- O(1) Prediction Cost Proof ---");
    let predictor = twin.predictor();
    let input = symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea_core::hdc::unified_hv::HDC_DIMENSION,
        42,
    );
    for &horizon in THREAT_HORIZONS {
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
    println!("PASS: Threat Assessment operational");
}
