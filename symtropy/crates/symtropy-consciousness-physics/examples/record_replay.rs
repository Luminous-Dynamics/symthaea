// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Record deterministic replay tapes for key experiments.
//!
//! Captures initial world state + per-tick consciousness data so reviewers
//! can reproduce results without running the full simulation.
//!
//! Run: cargo run --example record_replay -p symtropy-consciousness-physics --release
//!
//! Output: JSON replay files to stdout (pipe to file).

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::coupling::ConsciousnessField;
use symtropy_math::Point;
use symtropy_physics::{BodyHandle, PhysicsWorld};

const DT: f64 = 1.0 / 60.0;

/// Lightweight per-tick snapshot for replay verification.
struct TickSnapshot {
    tick: u64,
    positions: Vec<[f64; 2]>,
    phis: Vec<f64>,
    collective_phi: f64,
    alive: usize,
}

/// Run a reproducible experiment and collect snapshots.
fn run_experiment(
    name: &str,
    num_agents: usize,
    ticks: u64,
    phi_gravity: f64,
    inputs: &ConsciousnessInputs,
) -> Vec<TickSnapshot> {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut field = ConsciousnessField::<2>::new();
    field.phi_gravity_strength = phi_gravity;

    let mut handles = Vec::new();
    let r = 10.0f64;
    for i in 0..num_agents {
        let angle = std::f64::consts::TAU * i as f64 / num_agents as f64;
        let h = world.add_sphere(Point::new([r * angle.cos(), r * angle.sin()]), 1.0, 1.0);
        field.register(h, 100.0, 2.0);
        handles.push(h);
    }

    eprintln!("  Recording {name}: {num_agents} agents, {ticks} ticks, k={phi_gravity}");

    let mut snapshots = Vec::new();

    for tick in 0..ticks {
        // Update consciousness
        for &h in &handles {
            let pos = world
                .body(h)
                .map(|b| {
                    let p = b.position();
                    Point::new([p.coord(0), p.coord(1)])
                })
                .unwrap_or_else(Point::origin);
            field.update_entity(h, inputs, pos);
        }

        field.tick_prediction_errors();

        // Emotional contagion
        let positions: Vec<(BodyHandle, Point<2>)> = handles
            .iter()
            .map(|&h| {
                let pos = world
                    .body(h)
                    .map(|b| {
                        let p = b.position();
                        Point::new([p.coord(0), p.coord(1)])
                    })
                    .unwrap_or_else(Point::origin);
                (h, pos)
            })
            .collect();
        field.spread_emotional_contagion(&positions, DT);

        // Physics step
        world.step_with_callback(DT, &mut field);

        // Record snapshot every 10 ticks
        if tick % 10 == 0 {
            let pos_vec: Vec<[f64; 2]> = handles
                .iter()
                .map(|&h| {
                    world
                        .body(h)
                        .map(|b| {
                            let p = b.position();
                            [p.coord(0), p.coord(1)]
                        })
                        .unwrap_or([0.0, 0.0])
                })
                .collect();

            let phi_vec: Vec<f64> = handles
                .iter()
                .map(|&h| field.entities.get(&h).map(|e| e.phi()).unwrap_or(0.0))
                .collect();

            let collective = phi_vec.iter().sum::<f64>() / phi_vec.len().max(1) as f64;
            let alive = phi_vec.iter().filter(|&&p| p > 0.0).count();

            snapshots.push(TickSnapshot {
                tick,
                positions: pos_vec,
                phis: phi_vec,
                collective_phi: collective,
                alive,
            });
        }
    }

    snapshots
}

fn print_replay_json(name: &str, snapshots: &[TickSnapshot]) {
    // Minimal JSON output without serde
    println!("  {{");
    println!("    \"experiment\": \"{name}\",");
    println!("    \"num_snapshots\": {},", snapshots.len());

    if let Some(first) = snapshots.first() {
        println!(
            "    \"initial_collective_phi\": {:.6},",
            first.collective_phi
        );
    }
    if let Some(last) = snapshots.last() {
        println!("    \"final_collective_phi\": {:.6},", last.collective_phi);
        println!("    \"final_alive\": {},", last.alive);
        println!("    \"final_tick\": {},", last.tick);

        // Mean pairwise distance (clustering metric)
        let n = last.positions.len();
        if n > 1 {
            let mut total = 0.0;
            let mut pairs = 0u64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = last.positions[i][0] - last.positions[j][0];
                    let dy = last.positions[i][1] - last.positions[j][1];
                    total += (dx * dx + dy * dy).sqrt();
                    pairs += 1;
                }
            }
            println!("    \"final_clustering\": {:.4},", total / pairs as f64);
        }
    }

    // Summary trajectory (every 50th snapshot)
    println!("    \"phi_trajectory\": [");
    for (i, s) in snapshots.iter().enumerate().step_by(5) {
        let comma = if i + 5 < snapshots.len() { "," } else { "" };
        println!(
            "      {{\"tick\": {}, \"phi\": {:.6}, \"alive\": {}}}{comma}",
            s.tick, s.collective_phi, s.alive
        );
    }
    println!("    ]");
    println!("  }}");
}

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Symtropy Deterministic Replay Recorder");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let conscious_inputs = ConsciousnessInputs {
        phi: 0.5,
        broadcast: 0.5,
        working_memory: 0.5,
        attention: 0.5,
        recurrence: 0.5,
        embodiment: 0.5,
        knowledge: 0.5,
        synchrony: 0.5,
    };

    let unconscious_inputs = ConsciousnessInputs {
        phi: 0.0,
        broadcast: 0.0,
        working_memory: 0.0,
        attention: 0.0,
        recurrence: 0.0,
        embodiment: 0.0,
        knowledge: 0.0,
        synchrony: 0.0,
    };

    println!("{{");
    println!("  \"version\": \"1.0\",");
    println!("  \"engine\": \"symtropy-consciousness-physics\",");
    println!("  \"dt\": {DT},");
    println!("  \"experiments\": [");

    // Experiment 1: Conscious agents with coupling
    let snaps = run_experiment("conscious_coupled", 8, 600, 0.3, &conscious_inputs);
    print_replay_json("conscious_coupled", &snaps);
    println!(",");

    // Experiment 2: Unconscious agents (Φ=0)
    let snaps = run_experiment("unconscious", 8, 600, 0.3, &unconscious_inputs);
    print_replay_json("unconscious", &snaps);
    println!(",");

    // Experiment 3: No coupling (k=0)
    let snaps = run_experiment("decoupled", 8, 600, 0.0, &conscious_inputs);
    print_replay_json("decoupled", &snaps);
    println!(",");

    // Experiment 4: Strong coupling (phase transition territory)
    let snaps = run_experiment("strong_coupling", 8, 600, 0.8, &conscious_inputs);
    print_replay_json("strong_coupling", &snaps);

    println!("  ]");
    println!("}}");

    eprintln!("\n  Done. Pipe stdout to a JSON file for reproducibility.");
}
