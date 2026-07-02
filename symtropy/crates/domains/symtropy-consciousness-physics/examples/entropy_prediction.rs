// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thermodynamic Collapse Prediction — can available_work() forecast failure?
//!
//! Tracks entropy, temperature, and Helmholtz free energy (F = U - TS) for
//! each agent over time. Tests whether available_work() predicts collapse
//! before it happens — the first demonstration of genuine thermodynamic
//! early warning in a multi-agent consciousness system.
//!
//! Run: cargo run --example entropy_prediction --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 16;
const TICKS: usize = 8_000;
const DT: f64 = 1.0 / 64.0;
const NUM_SEEDS: usize = 10;

struct AgentSnapshot {
    tick: usize,
    agent: usize,
    energy: f64,
    entropy: f64,
    temperature: f64,
    available_work: f64,
    phi: f64,
    alive: bool,
    collapsed_within_500: bool, // label: does this agent collapse in next 500 ticks?
}

fn run_experiment(seed: u64) -> Vec<AgentSnapshot> {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    // Harsh constants — forces collapses for prediction data
    let mut constants = ThermodynamicConstants::research();
    constants.initial_energy = 80.0; // Very low runway
    constants.max_energy = 80.0;
    constants.consciousness_maintenance_per_tick = 0.40; // Double the research rate
    constants.harmony_resonance_regen_rate = 0.02; // Offloading barely helps
    constants.ambient_regen_rate = 0.001; // Almost no ambient regen
    consciousness.constants = constants;

    let wells = vec![SVector::from([30.0, 0.0])]; // Only ONE well
    let mut well_remaining = vec![2000.0f64]; // Small capacity

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 100.0;
        let y = (rng_f64(&mut rng) - 0.5) * 100.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.05;
        }
        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );
        if let Some(e) = consciousness.entities.get_mut(&h) {
            match i % 4 {
                0 => e.harmony_activations = [0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5],
                1 => e.harmony_activations = [0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.5],
                2 => e.harmony_activations = [0.5; 9],
                _ => e.harmony_activations = [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.3, 0.5],
            }
        }
        handles.push(h);
    }

    // First pass: run simulation and record collapse times
    let mut collapse_tick: Vec<Option<usize>> = vec![None; AGENTS];
    let mut snapshots_raw: Vec<(usize, usize, f64, f64, f64, f64, f64, bool)> = Vec::new();

    for tick in 0..TICKS {
        // Consciousness update
        for &h in &handles {
            let e = consciousness.entities.get(&h);
            let ef = e.map(|e| e.energy.fraction_remaining()).unwrap_or(0.0);
            let pe = e.map(|e| e.prediction_error).unwrap_or(0.0);
            let ht = e.map(|e| e.total_harmony_energy()).unwrap_or(0.0);
            let collapsed = e.map(|e| e.energy.is_collapsed()).unwrap_or(true);
            let inputs = if collapsed {
                ConsciousnessInputs {
                    phi: 0.0,
                    broadcast: 0.0,
                    working_memory: 0.0,
                    attention: 0.0,
                    recurrence: 0.0,
                    embodiment: 0.0,
                    knowledge: 0.0,
                    synchrony: 0.0,
                }
            } else {
                ConsciousnessInputs {
                    phi: ef,
                    broadcast: 0.5,
                    working_memory: (1.0 - pe).max(0.0),
                    attention: 0.5,
                    recurrence: 1.0,
                    embodiment: 0.7,
                    knowledge: (ht / 8.0).min(1.0),
                    synchrony: consciousness.collective_phi.max(0.5),
                }
            };
            consciousness.update_entity(h, &inputs, Point::origin());
        }

        // FEP gradient
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|&(_, &r)| r > 0.0)
            .map(|(&p, &r)| (p, (r / 4000.0).min(1.0)))
            .collect();
        for &h in &handles {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position();
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| (p - pos).norm() > 2.0)
                .cloned()
                .collect();
            let dir = fep_gradient::free_energy_gradient(
                &pos,
                e.energy.fraction_remaining(),
                &e.harmony_activations,
                &near,
                &wdata,
                None,
                0.0,
            );
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = dir * 25.0;
            }
        }

        // Maintenance + regen with well depletion
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let rm = consciousness.resource_regeneration_multiplier();
        let nw: Vec<Option<usize>> = handles
            .iter()
            .map(|&h| {
                world.body(h).and_then(|b| {
                    let pos = b.position();
                    wells
                        .iter()
                        .enumerate()
                        .find(|&(ref i, &w)| (pos - w).norm() < 35.0 && well_remaining[*i] > 0.0)
                        .map(|(i, _)| i)
                })
            })
            .collect();
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            let phi = consciousness.phi(h);
            consciousness.consume_energy(h, mr * (1.0 + phi * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                if let Some(wi) = nw[idx] {
                    let draw = wr.min(well_remaining[wi]);
                    e.energy.regenerate(draw);
                    well_remaining[wi] -= draw;
                }
            }
        }

        // Harmony offloading
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let (a, b) = match (
                    consciousness.entities.get(&ha),
                    consciousness.entities.get(&hb),
                ) {
                    (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                    _ => continue,
                };
                let res = HarmonyField::<2>::resonance(&a, &b);
                if res > 0.5 {
                    let rg =
                        consciousness.constants.harmony_resonance_regen_rate * (res - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.regenerate(rg);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.regenerate(rg);
                    }
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();

        // Record snapshots every 100 ticks
        if tick % 100 == 0 {
            for (idx, &h) in handles.iter().enumerate() {
                if let Some(e) = consciousness.entities.get(&h) {
                    let alive = !e.energy.is_collapsed();
                    if !alive && collapse_tick[idx].is_none() {
                        collapse_tick[idx] = Some(tick);
                    }
                    snapshots_raw.push((
                        tick,
                        idx,
                        e.energy.available,
                        e.energy.entropy,
                        e.energy.temperature,
                        e.energy.available_work(),
                        e.phi(),
                        alive,
                    ));
                }
            }
        }

        // Also record exact collapse ticks
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get(&h) {
                if e.energy.is_collapsed() && collapse_tick[idx].is_none() {
                    collapse_tick[idx] = Some(tick);
                }
            }
        }
    }

    // Label snapshots: did this agent collapse within the next 500 ticks?
    snapshots_raw
        .iter()
        .map(|&(tick, agent, energy, entropy, temp, aw, phi, alive)| {
            let will_collapse = collapse_tick[agent]
                .map(|ct| ct > tick && ct <= tick + 500)
                .unwrap_or(false);
            AgentSnapshot {
                tick,
                agent,
                energy,
                entropy,
                temperature: temp,
                available_work: aw,
                phi,
                alive,
                collapsed_within_500: will_collapse,
            }
        })
        .collect()
}

fn main() {
    eprintln!("=== Thermodynamic Collapse Prediction ===");
    eprintln!("Can available_work() predict agent failure?");
    eprintln!("Agents: {AGENTS}, Ticks: {TICKS}, Seeds: {NUM_SEEDS}");

    println!(
        "seed,tick,agent,energy,entropy,temperature,available_work,phi,alive,collapse_within_500"
    );

    let mut all_alive_aw: Vec<f64> = Vec::new(); // available_work for surviving agents
    let mut all_doomed_aw: Vec<f64> = Vec::new(); // available_work for agents about to collapse
    let mut total_collapsed = 0usize;
    let mut total_agents = 0usize;

    for s in 0..NUM_SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  Seed {seed}...");
        let snapshots = run_experiment(seed);

        let seed_collapsed = snapshots.iter().filter(|s| !s.alive).count();
        let seed_doomed = snapshots
            .iter()
            .filter(|s| s.collapsed_within_500 && s.alive)
            .count();

        for snap in &snapshots {
            println!(
                "{seed},{},{},{:.2},{:.6},{:.2},{:.2},{:.4},{},{}",
                snap.tick,
                snap.agent,
                snap.energy,
                snap.entropy,
                snap.temperature,
                snap.available_work,
                snap.phi,
                snap.alive,
                snap.collapsed_within_500
            );

            if snap.alive {
                if snap.collapsed_within_500 {
                    all_doomed_aw.push(snap.available_work);
                } else {
                    all_alive_aw.push(snap.available_work);
                }
            }
        }

        total_collapsed += seed_collapsed;
        total_agents += AGENTS;
    }

    // ROC analysis: available_work as predictor of collapse_within_500
    eprintln!("\n── Prediction Analysis ──");
    eprintln!(
        "  Total snapshots: alive={}, doomed={}",
        all_alive_aw.len(),
        all_doomed_aw.len()
    );
    eprintln!(
        "  Collapse rate: {:.1}%",
        total_collapsed as f64 / (total_agents * (TICKS / 100)) as f64 * 100.0
    );

    if !all_doomed_aw.is_empty() && !all_alive_aw.is_empty() {
        let mean_alive = all_alive_aw.iter().sum::<f64>() / all_alive_aw.len() as f64;
        let mean_doomed = all_doomed_aw.iter().sum::<f64>() / all_doomed_aw.len() as f64;
        eprintln!("  Mean available_work (surviving): {:.2} J", mean_alive);
        eprintln!("  Mean available_work (doomed):    {:.2} J", mean_doomed);
        eprintln!(
            "  Separation: {:.2} J ({:.1}x)",
            mean_alive - mean_doomed,
            mean_alive / mean_doomed.max(0.01)
        );

        // Simple threshold classifier
        let thresholds = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0];
        eprintln!("\n  Threshold  Sensitivity  Specificity  Accuracy");
        for &thresh in &thresholds {
            let tp = all_doomed_aw.iter().filter(|&&aw| aw < thresh).count() as f64;
            let fn_ = all_doomed_aw.iter().filter(|&&aw| aw >= thresh).count() as f64;
            let tn = all_alive_aw.iter().filter(|&&aw| aw >= thresh).count() as f64;
            let fp = all_alive_aw.iter().filter(|&&aw| aw < thresh).count() as f64;
            let sensitivity = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let specificity = if tn + fp > 0.0 { tn / (tn + fp) } else { 0.0 };
            let accuracy = if tp + tn + fp + fn_ > 0.0 {
                (tp + tn) / (tp + tn + fp + fn_)
            } else {
                0.0
            };
            eprintln!(
                "  {thresh:>6.0} J    {sensitivity:.3}        {specificity:.3}        {accuracy:.3}"
            );
        }

        use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
        let (u, z, p) = mann_whitney_u(&all_alive_aw, &all_doomed_aw);
        let d = cohens_d(&all_alive_aw, &all_doomed_aw);
        eprintln!("\n  Mann-Whitney U={:.0}, z={:.3}, p={:.4}", u, z, p);
        eprintln!(
            "  Cohen's d={:.3} ({})",
            d,
            if d.abs() > 0.8 {
                "large"
            } else if d.abs() > 0.5 {
                "medium"
            } else {
                "small"
            }
        );
        eprintln!(
            "  {}",
            if p < 0.05 {
                "SIGNIFICANT"
            } else {
                "not significant"
            }
        );
    } else {
        eprintln!("  Insufficient collapse events for analysis.");
        if all_doomed_aw.is_empty() {
            eprintln!("  (No agents collapsed within prediction window)");
        }
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
