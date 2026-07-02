// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PyPhi Validation: compare our softmin Φ against exact IIT Φ.
//!
//! Runs a 3-agent simulation, collects TPM via TpmCollector, then
//! calls PyPhi for ground-truth Φ. Reports the comparison.
//!
//! Requires: pyphi-validation feature + PyPhi Python environment.
//!
//! Run:
//!   cargo run --example pyphi_validate -p symtropy-consciousness-physics \
//!     --features pyphi-validation --release

#[cfg(not(feature = "pyphi-validation"))]
fn main() {
    eprintln!("This example requires --features pyphi-validation");
    std::process::exit(1);
}

#[cfg(feature = "pyphi-validation")]
fn main() {
    use nalgebra::SVector;
    use symthaea_consciousness_equation::ConsciousnessInputs;
    use symtropy_consciousness_physics::coupling::ConsciousnessField;
    use symtropy_consciousness_physics::pyphi_validation::{
        PHI_BINARY_THRESHOLD, TpmCollector, validate_against_pyphi,
    };
    use symtropy_math::Point;
    use symtropy_physics::PhysicsWorld;

    const NUM_AGENTS: usize = 3; // Small enough for exact PyPhi (2^3 = 8 states)
    const TICKS: u64 = 2000;
    const DT: f64 = 1.0 / 60.0;

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  PyPhi Validation: Softmin Φ vs Exact IIT Φ");
    eprintln!(
        "  {} agents, {} ticks, threshold={}",
        NUM_AGENTS, TICKS, PHI_BINARY_THRESHOLD
    );
    eprintln!("═══════════════════════════════════════════════════════════\n");

    // Create world
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut field = ConsciousnessField::<2>::new();
    field.phi_gravity_strength = 0.3;

    let mut handles = Vec::new();
    for i in 0..NUM_AGENTS {
        let angle = std::f64::consts::TAU * i as f64 / NUM_AGENTS as f64;
        let r = 8.0;
        let h = world.add_sphere(Point::new([r * angle.cos(), r * angle.sin()]), 1.0, 1.0);
        field.register(h, 100.0, 2.0);
        handles.push(h);
    }

    // Create TPM collector
    let mut collector = TpmCollector::new(handles.clone(), PHI_BINARY_THRESHOLD);

    // Vary inputs across ticks to explore state space
    let input_profiles = [
        ConsciousnessInputs {
            phi: 0.8,
            broadcast: 0.7,
            working_memory: 0.6,
            attention: 0.9,
            recurrence: 0.5,
            embodiment: 0.8,
            knowledge: 0.6,
            synchrony: 0.7,
        },
        ConsciousnessInputs {
            phi: 0.2,
            broadcast: 0.3,
            working_memory: 0.4,
            attention: 0.2,
            recurrence: 0.3,
            embodiment: 0.5,
            knowledge: 0.2,
            synchrony: 0.3,
        },
        ConsciousnessInputs {
            phi: 0.5,
            broadcast: 0.5,
            working_memory: 0.5,
            attention: 0.5,
            recurrence: 0.5,
            embodiment: 0.5,
            knowledge: 0.5,
            synchrony: 0.5,
        },
    ];

    eprintln!("  Simulating {} ticks...", TICKS);
    for tick in 0..TICKS {
        // Cycle through input profiles to explore different states
        let phase = (tick / 200) as usize % input_profiles.len();
        let inputs = &input_profiles[phase];

        // Vary per-agent to create asymmetry
        for (i, &h) in handles.iter().enumerate() {
            let mut agent_inputs = inputs.clone();
            // Add agent-specific variation
            agent_inputs.phi *= 1.0 - 0.2 * (i as f64 / NUM_AGENTS as f64);
            agent_inputs.attention *= 1.0 + 0.1 * (i as f64);

            let pos = world
                .body(h)
                .map(|b| {
                    let p = b.position();
                    Point::new([p[0], p[1]])
                })
                .unwrap_or_else(Point::origin);
            field.update_entity(h, &agent_inputs, pos);
        }

        field.tick_prediction_errors();

        // Emotional contagion
        let positions: Vec<_> = handles
            .iter()
            .map(|&h| {
                let pos = world
                    .body(h)
                    .map(|b| {
                        let p = b.position();
                        Point::new([p[0], p[1]])
                    })
                    .unwrap_or_else(Point::origin);
                (h, pos)
            })
            .collect();
        field.spread_emotional_contagion(&positions, DT);

        // Physics step
        world.step_with_callback(DT, &mut field);

        // Collect TPM data
        collector.tick(&field);

        // Progress
        if tick % 500 == 0 {
            eprintln!(
                "    tick {}: coverage={:.1}% ({}/{} states visited)",
                tick,
                collector.coverage() * 100.0,
                collector.tpm().states_visited(),
                collector.tpm().num_states(),
            );
        }
    }

    eprintln!("\n  TPM collection complete:");
    eprintln!("    Ticks recorded: {}", collector.ticks_recorded());
    eprintln!(
        "    State coverage: {:.1}% ({}/{})",
        collector.coverage() * 100.0,
        collector.tpm().states_visited(),
        collector.tpm().num_states(),
    );

    // Print the TPM for inspection
    let tpm = collector.tpm().to_state_by_node();
    eprintln!("\n  Empirical TPM (state-by-node):");
    for (idx, row) in tpm.iter().enumerate() {
        let state_bits: Vec<u8> = (0..NUM_AGENTS).map(|i| ((idx >> i) & 1) as u8).collect();
        let probs: Vec<String> = row.iter().map(|p| format!("{:.3}", p)).collect();
        eprintln!("    {:?} → [{}]", state_bits, probs.join(", "));
    }

    // Our engine's Φ values
    eprintln!("\n  Engine Φ values:");
    for &h in &handles {
        let phi = field.entities.get(&h).map(|e| e.phi()).unwrap_or(0.0);
        eprintln!("    Agent {}: Φ = {:.6}", h.0, phi);
    }

    // Call PyPhi
    eprintln!("\n  Calling PyPhi for exact IIT Φ...");
    let result = validate_against_pyphi(&collector, &field);

    eprintln!("\n═══════════════════════════════════════════════════════════");
    eprintln!("  Validation Results:");
    eprintln!("  Engine mean Φ:     {:.6}", result.symtropy_phi);
    if let Some(pp) = result.pyphi_phi {
        eprintln!("  PyPhi exact Φ:     {:.6}", pp);
        if let Some(err) = result.absolute_error {
            eprintln!("  Absolute error:    {:.6}", err);
        }
    }
    if let Some(ref e) = result.error {
        eprintln!("  Error: {}", e);
        eprintln!("  (Ensure PyPhi is installed: see scripts/pyphi_compute.py)");
    }
    eprintln!("  TPM coverage:      {:.1}%", result.tpm_coverage * 100.0);
    eprintln!("  Group size:        {}", result.group_size);
    eprintln!("═══════════════════════════════════════════════════════════");
}
