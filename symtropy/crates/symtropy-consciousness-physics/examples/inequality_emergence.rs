// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inequality Emergence — do thermodynamic agents develop wealth disparity?
//!
//! Tracks the Gini coefficient of energy distribution over time.
//! Tests whether cooperation reduces or amplifies inequality, and whether
//! initial position (proximity to wells) creates lasting advantage.
//!
//! Compares 3 conditions:
//! 1. EQUAL_START: All agents start at center with identical energy
//! 2. RANDOM_START: Random positions, identical energy
//! 3. UNEQUAL_START: Random positions, energy proportional to well proximity
//!
//! Piketty (2014) found r > g drives inequality in capitalism.
//! Our analogue: do agents near wells (r = return on position) accumulate
//! faster than cooperation can redistribute (g = growth from resonance)?
//!
//! Run: cargo run --example inequality_emergence --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::cohens_d;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 24;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 8;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy)]
enum StartCondition {
    Equal,
    Random,
    Unequal,
}

struct InequalityResult {
    condition: &'static str,
    alive: f64,
    gini_start: f64,
    gini_mid: f64,
    gini_end: f64,
    energy_top25: f64, // mean energy of top 25%
    energy_bot25: f64, // mean energy of bottom 25%
    mobility: f64,     // fraction of agents that changed quartile
}

fn gini(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    if mean < 1e-10 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += (values[i] - values[j]).abs();
        }
    }
    sum / (2.0 * n as f64 * n as f64 * mean)
}

fn run_experiment(start: StartCondition, seed: u64) -> InequalityResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![3000.0f64; 2];

    let mut rng = seed;
    let mut handles = Vec::new();
    let cond_name = match start {
        StartCondition::Equal => "EQUAL",
        StartCondition::Random => "RANDOM",
        StartCondition::Unequal => "UNEQUAL",
    };

    for i in 0..AGENTS {
        let (x, y) = match start {
            StartCondition::Equal => (0.0, (i as f64 - AGENTS as f64 / 2.0) * 3.0),
            _ => (
                (rng_f64(&mut rng) - 0.5) * 120.0,
                (rng_f64(&mut rng) - 0.5) * 120.0,
            ),
        };
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.05;
        }

        let start_energy = match start {
            StartCondition::Unequal => {
                // Energy proportional to proximity to nearest well
                let min_dist = wells
                    .iter()
                    .map(|w| ((x - w[0]).powi(2) + (y - w[1]).powi(2)).sqrt())
                    .fold(f64::MAX, f64::min);
                let frac = 1.0 - (min_dist / 100.0).min(1.0);
                50.0 + frac * 150.0 // 50-200J based on position
            }
            _ => consciousness.constants.initial_energy,
        };

        consciousness.register(
            h,
            consciousness.constants.max_energy,
            consciousness.constants.harmony_range,
        );
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.energy.available = start_energy;
            e.harmony_activations = HARMONY_PROFILES[i % 4];
        }
        handles.push(h);
    }

    // Record initial quartiles
    let initial_energies: Vec<f64> = handles
        .iter()
        .map(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| e.energy.available)
                .unwrap_or(0.0)
        })
        .collect();
    let gini_start = gini(&initial_energies);
    let initial_quartiles = quartile_assignments(&initial_energies);

    let mut gini_mid = 0.0;
    let mut mid_energies = vec![0.0f64; AGENTS];

    for tick in 0..TICKS {
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
            .map(|(&p, &r)| (p, (r / 3000.0).min(1.0)))
            .collect();
        for &h in &handles {
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
                }
            }
        }

        // Record midpoint
        if tick == TICKS / 2 {
            for (idx, &h) in handles.iter().enumerate() {
                mid_energies[idx] = consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.energy.available)
                    .unwrap_or(0.0);
            }
            gini_mid = gini(&mid_energies);
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    let final_energies: Vec<f64> = handles
        .iter()
        .map(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| e.energy.available)
                .unwrap_or(0.0)
        })
        .collect();
    let gini_end = gini(&final_energies);
    let final_quartiles = quartile_assignments(&final_energies);

    // Mobility: fraction that changed quartile
    let mobility = initial_quartiles
        .iter()
        .zip(final_quartiles.iter())
        .filter(|(a, b)| a != b)
        .count() as f64
        / AGENTS as f64;

    // Top/bottom 25% energy
    let mut sorted = final_energies.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let q = AGENTS / 4;
    let top25 = sorted[..q].iter().sum::<f64>() / q as f64;
    let bot25 = sorted[(AGENTS - q)..].iter().sum::<f64>() / q as f64;

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

    InequalityResult {
        condition: cond_name,
        alive,
        gini_start,
        gini_mid,
        gini_end,
        energy_top25: top25,
        energy_bot25: bot25,
        mobility,
    }
}

fn quartile_assignments(values: &[f64]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut assignments = vec![0usize; values.len()];
    let q = values.len() / 4;
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        assignments[idx] = rank / q.max(1);
    }
    assignments
}

fn main() {
    eprintln!("=== Inequality Emergence Experiment ===");
    eprintln!("Do thermodynamic agents develop wealth disparity?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    println!(
        "condition,seed,alive,gini_start,gini_mid,gini_end,top25_energy,bot25_energy,mobility"
    );

    let conditions = [
        (StartCondition::Equal, "EQUAL"),
        (StartCondition::Random, "RANDOM"),
        (StartCondition::Unequal, "UNEQUAL"),
    ];
    let mut all_results: Vec<(&str, Vec<InequalityResult>)> = Vec::new();

    for &(cond, name) in &conditions {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(cond, seed);
            println!(
                "{},{seed},{:.1},{:.4},{:.4},{:.4},{:.1},{:.1},{:.3}",
                r.condition,
                r.alive,
                r.gini_start,
                r.gini_mid,
                r.gini_end,
                r.energy_top25,
                r.energy_bot25,
                r.mobility
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: Gini {:.3}→{:.3}→{:.3}, top25={:.0}J, bot25={:.0}J, mobility={:.1}%",
            results.iter().map(|r| r.gini_start).sum::<f64>() / n,
            results.iter().map(|r| r.gini_mid).sum::<f64>() / n,
            results.iter().map(|r| r.gini_end).sum::<f64>() / n,
            results.iter().map(|r| r.energy_top25).sum::<f64>() / n,
            results.iter().map(|r| r.energy_bot25).sum::<f64>() / n,
            results.iter().map(|r| r.mobility).sum::<f64>() / n * 100.0
        );
        all_results.push((name, results));
    }

    eprintln!("\n── Inequality Dynamics ──");
    eprintln!("  Condition   Gini(t=0)  Gini(mid)  Gini(end)  Top25    Bot25    Mobility");
    for (name, results) in &all_results {
        let n = results.len() as f64;
        eprintln!(
            "  {:10} {:8.3}    {:8.3}    {:8.3}    {:5.0}J   {:5.0}J   {:5.1}%",
            name,
            results.iter().map(|r| r.gini_start).sum::<f64>() / n,
            results.iter().map(|r| r.gini_mid).sum::<f64>() / n,
            results.iter().map(|r| r.gini_end).sum::<f64>() / n,
            results.iter().map(|r| r.energy_top25).sum::<f64>() / n,
            results.iter().map(|r| r.energy_bot25).sum::<f64>() / n,
            results.iter().map(|r| r.mobility).sum::<f64>() / n * 100.0
        );
    }

    // Key question: does cooperation reduce inequality?
    if all_results.len() >= 2 {
        let equal_gini: Vec<f64> = all_results[0].1.iter().map(|r| r.gini_end).collect();
        let unequal_gini: Vec<f64> = all_results[2].1.iter().map(|r| r.gini_end).collect();
        let d = cohens_d(&equal_gini, &unequal_gini);
        eprintln!("\n── EQUAL vs UNEQUAL start → final Gini ──");
        eprintln!(
            "  Cohen's d={d:.3} ({})",
            if d.abs() > 0.8 {
                "large"
            } else if d.abs() > 0.5 {
                "medium"
            } else {
                "small"
            }
        );
        eprintln!(
            "  {} initial conditions → {} final inequality",
            if d.abs() < 0.3 {
                "Different"
            } else {
                "Different"
            },
            if d.abs() < 0.3 { "SAME" } else { "different" }
        );
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
