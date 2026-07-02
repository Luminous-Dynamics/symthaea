// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generate multiple TPMs for PyPhi correlation sweep.
//!
//! Runs 3-agent simulations with varying parameters, collects TPMs,
//! outputs JSONL for the pyphi_sweep.py script.
//!
//! Run: cargo run --example generate_tpms -p symtropy-consciousness-physics \
//!        --features pyphi-validation --release | python3 scripts/pyphi_sweep.py

#[cfg(not(feature = "pyphi-validation"))]
fn main() {
    eprintln!("Requires --features pyphi-validation");
    std::process::exit(1);
}

#[cfg(feature = "pyphi-validation")]
fn main() {
    use nalgebra::SVector;
    use symthaea_consciousness_equation::ConsciousnessInputs;
    use symtropy_consciousness_physics::coupling::ConsciousnessField;
    use symtropy_consciousness_physics::pyphi_validation::{PHI_BINARY_THRESHOLD, TpmCollector};
    use symtropy_math::Point;
    use symtropy_physics::PhysicsWorld;

    const N: usize = 3;
    const TICKS: u64 = 1500;
    const DT: f64 = 1.0 / 60.0;

    // Sweep across different input configurations
    let configs: Vec<(&str, f64, f64, f64)> = vec![
        ("low_phi", 0.1, 0.1, 0.0),
        ("med_low", 0.3, 0.3, 0.1),
        ("medium", 0.5, 0.5, 0.3),
        ("med_high", 0.7, 0.7, 0.5),
        ("high_phi", 0.9, 0.9, 0.8),
        ("asymmetric", 0.8, 0.2, 0.5),
        ("max_sync", 0.5, 0.5, 1.0),
        ("min_sync", 0.5, 0.5, 0.0),
        ("high_k", 0.5, 0.5, 0.3), // different phi_gravity
        ("zero_k", 0.5, 0.5, 0.3),
    ];

    eprintln!(
        "Generating TPMs for {} configurations, {} agents, {} ticks each...\n",
        configs.len(),
        N,
        TICKS
    );

    for (label, phi_base, attention, synchrony) in &configs {
        let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
        let mut field = ConsciousnessField::<2>::new();

        // Vary phi_gravity for the last two configs
        field.phi_gravity_strength = if *label == "high_k" {
            0.8
        } else if *label == "zero_k" {
            0.0
        } else {
            0.3
        };

        let mut handles = Vec::new();
        for i in 0..N {
            let angle = std::f64::consts::TAU * i as f64 / N as f64;
            let h = world.add_sphere(Point::new([8.0 * angle.cos(), 8.0 * angle.sin()]), 1.0, 1.0);
            field.register(h, 100.0, 2.0);
            handles.push(h);
        }

        let mut collector = TpmCollector::new(handles.clone(), PHI_BINARY_THRESHOLD);

        // Run simulation with varying inputs per agent
        for tick in 0..TICKS {
            let phase = (tick as f64 / 300.0).sin() * 0.2;
            for (i, &h) in handles.iter().enumerate() {
                let agent_phi = phi_base + phase * (i as f64 / N as f64);
                let inputs = ConsciousnessInputs {
                    phi: agent_phi.clamp(0.0, 1.0),
                    broadcast: *phi_base * 0.8,
                    working_memory: *phi_base * 0.7,
                    attention: *attention + 0.1 * (i as f64),
                    recurrence: *phi_base * 0.6,
                    embodiment: 0.8,
                    knowledge: *phi_base * 0.5,
                    synchrony: *synchrony,
                };
                let pos = world
                    .body(h)
                    .map(|b| {
                        let p = b.position();
                        Point::new([p[0], p[1]])
                    })
                    .unwrap_or_else(Point::origin);
                field.update_entity(h, &inputs, pos);
            }
            field.tick_prediction_errors();
            world.step_with_callback(DT, &mut field);
            collector.tick(&field);
        }

        // Get engine's mean Phi
        let engine_phi: f64 = handles
            .iter()
            .filter_map(|h| field.entities.get(h).map(|e| e.phi()))
            .sum::<f64>()
            / N as f64;

        // Get current binary state
        let current_state: Vec<u8> = handles
            .iter()
            .map(|h| {
                field
                    .entities
                    .get(h)
                    .map(|e| {
                        if e.phi() >= PHI_BINARY_THRESHOLD {
                            1
                        } else {
                            0
                        }
                    })
                    .unwrap_or(0)
            })
            .collect();

        // Build TPM
        let tpm = collector.tpm().to_state_by_node();

        // Output JSONL
        let tpm_str: Vec<String> = tpm
            .iter()
            .map(|row| {
                let vals: Vec<String> = row.iter().map(|v| format!("{v:.6}")).collect();
                format!("[{}]", vals.join(","))
            })
            .collect();
        let state_str: Vec<String> = current_state.iter().map(|s| format!("{s}")).collect();

        println!(
            r#"{{"label":"{}","engine_phi":{:.6},"tpm":[{}],"state":[{}],"coverage":{:.2},"ticks":{}}}"#,
            label,
            engine_phi,
            tpm_str.join(","),
            state_str.join(","),
            collector.coverage(),
            collector.ticks_recorded(),
        );

        eprintln!(
            "  {}: engine_phi={:.4}, coverage={:.0}%",
            label,
            engine_phi,
            collector.coverage() * 100.0
        );
    }
}
