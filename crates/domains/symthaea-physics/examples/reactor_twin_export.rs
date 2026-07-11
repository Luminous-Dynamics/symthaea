// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exports one run of the FissionTwin degradation scenario (the same
//! scenario as `fission_dispatch_demo.rs`) as JSON on stdout, for replay
//! in the `sol-atlas-leptos` "Reactor Digital Twin" demo page.
//!
//! This is a build-time data generator, not a runtime service: the output
//! is a static snapshot checked in as `sol-atlas-leptos/assets/data/
//! reactor-twin-demo.json` and replayed client-side. See that page's own
//! disclaimer for why this isn't presented as live telemetry.
//!
//! Run: `cargo run --release --example reactor_twin_export -p symthaea-physics`

use serde::Serialize;
use symthaea_physics::{
    FissionFepAction, FissionOutput, FissionReading, FissionSafetyLevel, FissionTwin,
    SensorNodeRegistration, simulated_dispatch_for_degradation,
};

#[derive(Serialize)]
struct ExportedTick {
    t_seconds: u32,
    power_output: f64,
    coolant_temp: f64,
    neutron_flux: f64,
    pressure: f64,
    control_rod_pos: f64,
    free_energy: f64,
    confidence: f64,
    safety_level: String,
    /// True only for the hand-built capstone tick (see module docs) --
    /// this run's real, steady-state scenario never reaches Orange/Red.
    synthetic: bool,
    dispatch_order: Option<ExportedDispatchOrder>,
}

#[derive(Serialize)]
struct ExportedDispatchOrder {
    priority_label: String,
    description: String,
}

fn healthy() -> FissionReading {
    FissionReading {
        power_output: 0.8,
        coolant_temp: 300.0,
        neutron_flux: 0.5,
        pressure: 10.0,
        control_rod_pos: 0.5,
    }
}

fn degrading(step: usize, ramp_steps: usize) -> FissionReading {
    let t = (step as f64 / ramp_steps as f64).min(1.0);
    FissionReading {
        power_output: 0.8 - 0.78 * t,
        coolant_temp: 300.0 + 100.0 * t,
        neutron_flux: 0.5 + 0.5 * t,
        pressure: 10.0 + 5.0 * t,
        control_rod_pos: 0.5 * (1.0 - t),
    }
}

fn main() {
    let sensor =
        SensorNodeRegistration::fission_twin("plant-alpha-core-monitor", 34.0522, -118.2437);
    let mut twin = FissionTwin::new();
    twin.set_reference(&healthy());

    const STEPS: usize = 200;
    const RAMP_STEPS: usize = 40;
    let mut ticks = Vec::with_capacity(STEPS + 1);
    let mut last_t = 0u32;

    for step in 0..STEPS {
        let reading = degrading(step, RAMP_STEPS);
        let output = twin.step(&reading, 10.0);
        let t_seconds = (step * 10) as u32;
        last_t = t_seconds;
        ticks.push(ExportedTick {
            t_seconds,
            power_output: reading.power_output,
            coolant_temp: reading.coolant_temp,
            neutron_flux: reading.neutron_flux,
            pressure: reading.pressure,
            control_rod_pos: reading.control_rod_pos,
            free_energy: output.free_energy,
            confidence: (1.0 - output.free_energy).clamp(0.0, 1.0),
            safety_level: format!("{:?}", output.safety_level),
            synthetic: false,
            dispatch_order: None,
        });
    }

    // Synthetic capstone tick: this steady-state scenario's free_energy
    // plateaus at ~0.24, well under the 0.5 Orange threshold (see
    // NUCLEAR_ENERGY_PLAN_2026-07-06.md Phase 4's finding). Hand-build one
    // Orange FissionOutput so the replay can still show the dispatch-order
    // path -- explicitly flagged `synthetic: true`, not something the twin
    // itself produced from this scenario.
    let synthetic_output = FissionOutput {
        free_energy: 0.62,
        recommended_action: FissionFepAction::ReducePower,
        safety_level: FissionSafetyLevel::Orange,
        prediction_similarities: vec![],
    };
    let order = simulated_dispatch_for_degradation(&sensor, &synthetic_output)
        .expect("Orange must always produce a dispatch order");
    ticks.push(ExportedTick {
        t_seconds: last_t + 10,
        power_output: 0.02,
        coolant_temp: 400.0,
        neutron_flux: 1.0,
        pressure: 15.0,
        control_rod_pos: 0.0,
        free_energy: synthetic_output.free_energy,
        confidence: (1.0 - synthetic_output.free_energy).clamp(0.0, 1.0),
        safety_level: "Orange".to_string(),
        synthetic: true,
        dispatch_order: Some(ExportedDispatchOrder {
            priority_label: order.priority_label.to_string(),
            description: order.description,
        }),
    });

    println!("{}", serde_json::to_string_pretty(&ticks).unwrap());
}
