// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end demo of the FissionTwin → robotics-dispatch pipeline shape.
//!
//! Per `symthaea/NUCLEAR_ENERGY_PLAN_2026-07-06.md` Phase 4: "FissionTwin
//! flags degradation → dispatch order for manipulator rad-zone inspection
//! → report to `mycelix-energy` grid zome. All simulated; the point is the
//! pipeline shape." This example covers the first two steps. The third
//! (mycelix-energy grid zome report) is deliberately not implemented here
//! — see `crate::fission_dispatch`'s module docs for why.
//!
//! Run: `cargo run --example fission_dispatch_demo -p symthaea-physics`

use symthaea_physics::{
    FissionFepAction, FissionOutput, FissionReading, FissionSafetyLevel, FissionTelemetryPayload,
    FissionTwin, SensorNodeRegistration, simulated_dispatch_for_degradation,
};

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
    // Ramp to a near-clamp-extreme reading over `ramp_steps`, then HOLD
    // there. A held-constant off-reference reading's free energy
    // converges to a *fixed* similarity gap rather than climbing forever
    // (confirmed empirically while building this demo) — so reaching
    // Orange/Red at all requires pushing every channel toward its
    // physical extreme, not just "somewhat worse."
    let t = (step as f64 / ramp_steps as f64).min(1.0);
    FissionReading {
        power_output: 0.8 - 0.78 * t,     // 0.8 -> 0.02 (near-zero power)
        coolant_temp: 300.0 + 100.0 * t,  // 300 -> 400 (clamp)
        neutron_flux: 0.5 + 0.5 * t,      // 0.5 -> 1.0 (clamp)
        pressure: 10.0 + 5.0 * t,         // 10 -> 15 (clamp)
        control_rod_pos: 0.5 * (1.0 - t), // 0.5 -> 0 (fully withdrawn)
    }
}

fn main() {
    println!("=== FissionTwin -> robotics-dispatch pipeline shape demo ===");
    println!("Advisory-only, non-1E monitoring. Nothing here is a safety-grade");
    println!("reactor protection function, and nothing calls a live conductor.\n");

    let sensor =
        SensorNodeRegistration::fission_twin("plant-alpha-core-monitor", 34.0522, -118.2437);
    println!("1. SensorNode registration (mirrors RoboticAsset):");
    println!("   {sensor:#?}\n");

    let mut twin = FissionTwin::new();
    let reference = healthy();
    twin.set_reference(&reference);

    println!("2. Telemetry stream (mirrors TelemetryReport), printed every 10th tick:");
    const STEPS: usize = 200;
    const RAMP_STEPS: usize = 40;
    let mut max_free_energy = 0.0_f64;
    for step in 0..STEPS {
        let reading = degrading(step, RAMP_STEPS);
        let output = twin.step(&reading, 10.0);
        let payload = FissionTelemetryPayload::from_fission_output(
            &reading,
            &output,
            step as i64 * 10_000_000,
            sensor.location_lat,
            sensor.location_lon,
        );
        max_free_energy = max_free_energy.max(output.free_energy);

        if step % 10 == 0 {
            println!(
                "   t={:>4}s  safety={:<7} confidence={:.3}  free_energy={:.3}",
                step * 10,
                payload.safety_level,
                payload.consciousness_level,
                output.free_energy,
            );
        }

        if let Some(order) = simulated_dispatch_for_degradation(&sensor, &output) {
            println!("\n3. Simulated dispatch order (mirrors DispatchOrder), NOT sent anywhere:");
            println!("   {order:#?}\n");
            print_report_deferred_note();
            return;
        }
    }

    // Honest finding, discovered while building this demo, not swept
    // under the rug: pushed every channel to its physical clamp extreme
    // (near-zero power, max coolant temp/flux/pressure, fully-withdrawn
    // rods) and held it there — free energy still plateaus at
    // {max_free_energy:.3}, well below the Orange threshold (0.5). This
    // 5-channel FissionTwin's free-energy metric is a fixed similarity
    // gap between two HDC encodings; a smooth/steady departure from
    // reference has a real, apparently intrinsic ceiling for this
    // encoder that Orange/Red don't reach. (Contrast the NPPAD
    // validation: Orange/Red-equivalent escalation there came from an
    // ABRUPT jump — a reactor scram's own aftermath — not a steady
    // state. Consistent with, not contradicting, that finding.)
    println!(
        "\n(Ceiling reached: max free_energy={max_free_energy:.3}, never crossed the 0.5 \
         Orange threshold even at physical clamp extremes — see the doc comment above this \
         message in fission_dispatch_demo.rs for what that means. Continuing with a \
         synthetic Orange result below to still demonstrate the dispatch-order code path.)\n"
    );

    // Bonus: hand-construct an Orange FissionOutput to demonstrate what
    // the downstream dispatch-order generation actually does — this is
    // legitimate because that logic only depends on FissionOutput's
    // fields, not on how a real FissionTwin would reach them.
    let synthetic_output = FissionOutput {
        free_energy: 0.62,
        recommended_action: FissionFepAction::ReducePower,
        safety_level: FissionSafetyLevel::Orange,
        prediction_similarities: vec![],
    };
    println!("3. Synthetic Orange result -> simulated dispatch order (mirrors DispatchOrder):");
    let order = simulated_dispatch_for_degradation(&sensor, &synthetic_output)
        .expect("Orange must always produce a dispatch order");
    println!("   {order:#?}\n");
    print_report_deferred_note();
}

fn print_report_deferred_note() {
    println!(
        "4. Report to mycelix-energy grid zome: DEFERRED.\n\
         \x20  mycelix-energy has active concurrent work in this monorepo as of 2026-07-09\n\
         \x20  (see MYCELIX_AUTHOR_BINDING_TRIAGE_2026-07-09.md's candidate list) -- this\n\
         \x20  demo intentionally stops at the dispatch-order shape rather than risk\n\
         \x20  touching that cluster."
    );
}
