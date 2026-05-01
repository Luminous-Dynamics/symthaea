// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-agent formation flight example.
//!
//! Trains 3 quadrotors in line formation, each with independent HDC-LTC controllers
//! and a shared `FormationController` for setpoint coordination.
//!
//! Run: `cargo run --features multirotor --example quadrotor_formation --release`

use symthaea::multirotor::formation::{FormationController, FormationShape, FormationState};
use symthaea::multirotor::{
    pd_baseline, FlightConfig, FlightController, FlightSetpoint, PdGains, PhysicsSimulator,
    QuadrotorHdcEncoder, SimplePhysicsSimulator,
};
use symthaea::symthaea_core::genesis::GenesisSeed;

const N_AGENTS: usize = 3;
const STEPS: usize = 1000;
const DT: f64 = 0.002;
const TRAIN_EVERY: usize = 4;

fn main() {
    println!("=== Symthaea Flight: Formation Flight (3 agents, line) ===");
    println!();

    let config = FlightConfig::default();
    let pd_gains = PdGains::default();
    let shape = FormationShape::Line { spacing: 0.3 };

    // Initialize per-agent state
    let mut physics: Vec<SimplePhysicsSimulator> = (0..N_AGENTS)
        .map(|_| SimplePhysicsSimulator::new())
        .collect();

    let mut encoders: Vec<QuadrotorHdcEncoder> = (0..N_AGENTS)
        .map(|i| {
            let genesis = GenesisSeed::from_phrase(&format!("formation-agent-{i}"));
            QuadrotorHdcEncoder::new(&genesis, config.num_levels)
        })
        .collect();

    let mut controllers: Vec<FlightController> = (0..N_AGENTS)
        .map(|i| {
            let genesis = GenesisSeed::from_phrase(&format!("formation-agent-{i}"));
            FlightController::new(&genesis, &config)
        })
        .collect();

    let mut formation_ctrls: Vec<FormationController> = (0..N_AGENTS)
        .map(|i| FormationController::new(i, shape.clone(), N_AGENTS))
        .collect();

    // Set formation center at hover altitude
    for fc in &mut formation_ctrls {
        fc.set_center([0.0, 0.0, 0.1]);
    }

    println!("Formation: Line with 0.3m spacing");
    println!("Agents: {N_AGENTS}");
    println!(
        "Steps: {STEPS} ({:.1}s at {:.0}Hz)",
        STEPS as f64 * DT,
        1.0 / DT
    );
    println!();

    // Print formation setpoints
    println!("Target positions:");
    for i in 0..N_AGENTS {
        let sp = formation_ctrls[i].compute_setpoint();
        println!(
            "  Agent {i}: [{:.2}, {:.2}, {:.2}]",
            sp.offset[0], sp.offset[1], sp.offset[2]
        );
    }
    println!();

    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10}",
        "Step", "Agent 0", "Agent 1", "Agent 2", "Max Err"
    );
    println!("{}", "-".repeat(50));

    for step in 0..STEPS {
        // Share formation state between agents
        let states: Vec<FormationState> = (0..N_AGENTS)
            .map(|i| FormationState {
                agent_id: i,
                encoded_state: vec![],
                position: physics[i].state().position,
                timestamp: physics[i].state().timestamp,
            })
            .collect();

        for i in 0..N_AGENTS {
            for s in &states {
                formation_ctrls[i].update_neighbor(s.clone());
            }
        }

        // Run each agent's control loop
        let mut errors = [0.0f64; N_AGENTS];
        for i in 0..N_AGENTS {
            let sp = formation_ctrls[i].compute_setpoint();
            let agent_setpoint = FlightSetpoint {
                position: sp.offset,
                yaw: 0.0,
            };

            let state = physics[i].state().clone();
            let sensor_hv = encoders[i].encode(&state);
            let command = controllers[i].forward(&sensor_hv, DT as f32);

            // Train toward PD baseline for this agent's formation setpoint
            if step % TRAIN_EVERY == 0 {
                let target = pd_baseline(&state, &agent_setpoint, &pd_gains);
                controllers[i].train_step(&sensor_hv, &target, DT as f32, None);
            }

            physics[i].step(&command, DT);
            errors[i] = formation_ctrls[i].formation_error(&physics[i].state().position);
        }

        // Print progress at intervals
        if step % 100 == 0 || step == STEPS - 1 {
            let max_err = errors.iter().cloned().fold(0.0f64, f64::max);
            println!(
                "{:>5} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                step, errors[0], errors[1], errors[2], max_err,
            );
        }
    }

    println!();

    // Final formation accuracy
    let mut total_err = 0.0;
    for i in 0..N_AGENTS {
        let err = formation_ctrls[i].formation_error(&physics[i].state().position);
        total_err += err;
        println!(
            "Agent {i} final position: [{:.4}, {:.4}, {:.4}] (error: {:.4}m)",
            physics[i].state().position[0],
            physics[i].state().position[1],
            physics[i].state().position[2],
            err
        );
    }
    println!();
    println!("Mean formation error: {:.4}m", total_err / N_AGENTS as f64);
}
