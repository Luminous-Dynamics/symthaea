// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Learning Cooperation — does reward-modulated plasticity change cooperation?
//!
//! Compares agents with fixed FEP gradient weights (current baseline) vs
//! agents that learn their FEP weights via reward-modulated plasticity.
//! The reward signal is energy change: positive when cooperation succeeds,
//! negative when the agent loses energy.
//!
//! This is the critical experiment: do agents that LEARN their behavioral
//! weights cooperate differently than agents with hardcoded weights?
//!
//! Run: cargo run --example learning_cooperation --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient::{self, LearnedFepWeights};
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20; // proper statistical power

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

struct LearnResult {
    condition: &'static str,
    alive: f64,
    energy: f64,
    clustering: f64,
    cooperation: f64,
    mean_weight_drift: f64,
    final_coop_weight: f64,
    final_well_weight: f64,
    final_danger_weight: f64,
}

fn run_experiment(use_learning: bool, seed: u64) -> LearnResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    consciousness.constants = constants;

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![2500.0f64; 2];

    let mut rng = seed;
    let mut handles = Vec::new();
    let mut agent_weights: Vec<LearnedFepWeights> = Vec::new();

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
            e.harmony_activations = HARMONY_PROFILES[i % 4];
        }
        handles.push(h);
        agent_weights.push(LearnedFepWeights::default());
    }

    let mut cooperation_events = 0u64;
    let cond_name = if use_learning { "LEARNING" } else { "FIXED" };

    for _tick in 0..TICKS {
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

        // FEP gradient (with or without learning)
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position().0,
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|(_, &r)| r > 0.0)
            .map(|(&p, &r)| (p, (r / 2500.0).min(1.0)))
            .collect();

        for (idx, &h) in handles.iter().enumerate() {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position().0;
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();

            if use_learning {
                let (dir, contributions) = fep_gradient::free_energy_gradient_learned(
                    &pos,
                    e.energy.fraction_remaining(),
                    None,
                    &e.harmony_activations,
                    &near,
                    &wdata,
                    None,
                    0.0,
                    &agent_weights[idx],
                );
                // Update weights with reward signal
                let current_energy = e.energy.fraction_remaining();
                agent_weights[idx].update(current_energy, contributions);
                if let Some(b) = world.body_mut(h) {
                    b.linear_velocity = dir * 20.0;
                }
            } else {
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
                    b.linear_velocity = dir * 20.0;
                }
            }
        }

        // Maintenance + regen
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let nw: Vec<Option<usize>> = handles
            .iter()
            .map(|&h| {
                world.body(h).and_then(|b| {
                    let pos = b.position().0;
                    wells
                        .iter()
                        .enumerate()
                        .find(|(i, &w)| (pos - w).norm() < 35.0 && well_remaining[*i] > 0.0)
                        .map(|(i, _)| i)
                })
            })
            .collect();
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                if let Some(wi) = nw[idx] {
                    let d = wr.min(well_remaining[wi]);
                    e.energy.regenerate(d);
                    well_remaining[wi] -= d;
                }
            }
        }

        // Cooperation
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let in_range = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => {
                        a.position().distance(b.position()) < consciousness.constants.harmony_range
                    }
                    _ => false,
                };
                if !in_range {
                    continue;
                }
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
                    cooperation_events += 1;
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    let alive = handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count() as f64;
    let energy = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;
    let clustering = avg_nn(&world, &handles);

    // Weight analysis
    let mean_drift = agent_weights
        .iter()
        .map(|w| w.drift_from_default())
        .sum::<f64>()
        / AGENTS as f64;
    let mean_coop = agent_weights.iter().map(|w| w.cooperation).sum::<f64>() / AGENTS as f64;
    let mean_well = agent_weights.iter().map(|w| w.energy_well).sum::<f64>() / AGENTS as f64;
    let mean_danger = agent_weights.iter().map(|w| w.danger).sum::<f64>() / AGENTS as f64;

    LearnResult {
        condition: cond_name,
        alive,
        energy,
        clustering,
        cooperation: cooperation_events as f64,
        mean_weight_drift: mean_drift,
        final_coop_weight: mean_coop,
        final_well_weight: mean_well,
        final_danger_weight: mean_danger,
    }
}

fn main() {
    eprintln!("=== Learning Cooperation Experiment ===");
    eprintln!("Does reward-modulated plasticity change cooperation?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds (proper power)");

    println!(
        "condition,seed,alive,energy,clustering,cooperation,weight_drift,coop_w,well_w,danger_w"
    );

    let mut fixed_results = Vec::new();
    let mut learn_results = Vec::new();

    for s in 0..SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  FIXED seed={seed}...");
        let r = run_experiment(false, seed);
        println!(
            "{},{seed},{:.1},{:.1},{:.2},{:.0},{:.4},{:.2},{:.2},{:.2}",
            r.condition,
            r.alive,
            r.energy,
            r.clustering,
            r.cooperation,
            r.mean_weight_drift,
            r.final_coop_weight,
            r.final_well_weight,
            r.final_danger_weight
        );
        fixed_results.push(r);
    }
    for s in 0..SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  LEARNING seed={seed}...");
        let r = run_experiment(true, seed);
        println!(
            "{},{seed},{:.1},{:.1},{:.2},{:.0},{:.4},{:.2},{:.2},{:.2}",
            r.condition,
            r.alive,
            r.energy,
            r.clustering,
            r.cooperation,
            r.mean_weight_drift,
            r.final_coop_weight,
            r.final_well_weight,
            r.final_danger_weight
        );
        learn_results.push(r);
    }

    let n = SEEDS as f64;
    eprintln!("\n── FIXED (baseline) ──");
    eprintln!(
        "  Alive:       {:.1}/{AGENTS}",
        fixed_results.iter().map(|r| r.alive).sum::<f64>() / n
    );
    eprintln!(
        "  Energy:      {:.1} J",
        fixed_results.iter().map(|r| r.energy).sum::<f64>() / n
    );
    eprintln!(
        "  Clustering:  {:.2}",
        fixed_results.iter().map(|r| r.clustering).sum::<f64>() / n
    );
    eprintln!(
        "  Cooperation: {:.0}",
        fixed_results.iter().map(|r| r.cooperation).sum::<f64>() / n
    );

    eprintln!("\n── LEARNING ──");
    eprintln!(
        "  Alive:       {:.1}/{AGENTS}",
        learn_results.iter().map(|r| r.alive).sum::<f64>() / n
    );
    eprintln!(
        "  Energy:      {:.1} J",
        learn_results.iter().map(|r| r.energy).sum::<f64>() / n
    );
    eprintln!(
        "  Clustering:  {:.2}",
        learn_results.iter().map(|r| r.clustering).sum::<f64>() / n
    );
    eprintln!(
        "  Cooperation: {:.0}",
        learn_results.iter().map(|r| r.cooperation).sum::<f64>() / n
    );
    eprintln!(
        "  Weight drift: {:.4}",
        learn_results
            .iter()
            .map(|r| r.mean_weight_drift)
            .sum::<f64>()
            / n
    );
    eprintln!(
        "  Learned coop:   {:.3} (default: 2.0)",
        learn_results
            .iter()
            .map(|r| r.final_coop_weight)
            .sum::<f64>()
            / n
    );
    eprintln!(
        "  Learned well:   {:.3} (default: 4.0)",
        learn_results
            .iter()
            .map(|r| r.final_well_weight)
            .sum::<f64>()
            / n
    );
    eprintln!(
        "  Learned danger: {:.3} (default: 3.0)",
        learn_results
            .iter()
            .map(|r| r.final_danger_weight)
            .sum::<f64>()
            / n
    );

    // Statistical tests with Holm-Bonferroni correction
    let f_alive: Vec<f64> = fixed_results.iter().map(|r| r.alive).collect();
    let l_alive: Vec<f64> = learn_results.iter().map(|r| r.alive).collect();
    let (_, _, p_alive) = mann_whitney_u(&f_alive, &l_alive);
    let d_alive = cohens_d(&f_alive, &l_alive);

    let f_coop: Vec<f64> = fixed_results.iter().map(|r| r.cooperation).collect();
    let l_coop: Vec<f64> = learn_results.iter().map(|r| r.cooperation).collect();
    let (_, _, p_coop) = mann_whitney_u(&f_coop, &l_coop);
    let d_coop = cohens_d(&f_coop, &l_coop);

    let f_clust: Vec<f64> = fixed_results.iter().map(|r| r.clustering).collect();
    let l_clust: Vec<f64> = learn_results.iter().map(|r| r.clustering).collect();
    let (_, _, p_clust) = mann_whitney_u(&f_clust, &l_clust);
    let d_clust = cohens_d(&f_clust, &l_clust);

    // Holm-Bonferroni correction for 3 comparisons
    let tests = vec![
        ("survival", p_alive),
        ("cooperation", p_coop),
        ("clustering", p_clust),
    ];
    let corrected = holm_bonferroni(&tests, 0.05);

    eprintln!("\n── FIXED vs LEARNING (Holm-Bonferroni corrected, k=3) ──");
    let effects = [d_alive, d_coop, d_clust];
    for (i, &(label, adj_p, sig)) in corrected.iter().enumerate() {
        let d = effects[i];
        let size = if d.abs() > 0.8 {
            "large"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        };
        eprintln!(
            "  {label:12}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
            if sig { "← SIGNIFICANT" } else { "" }
        );
    }

    eprintln!("\n=== Complete ===");
}

fn avg_nn(world: &PhysicsWorld<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|h| world.body(*h).map(|b| b.position().0))
        .collect();
    if pos.len() < 2 {
        return f64::MAX;
    }
    pos.iter()
        .enumerate()
        .map(|(i, p)| {
            pos.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, q)| (p - q).norm())
                .fold(f64::MAX, f64::min)
        })
        .sum::<f64>()
        / pos.len() as f64
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
