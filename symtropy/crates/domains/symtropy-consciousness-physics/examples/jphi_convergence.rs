// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! J/Phi Convergence Experiment — does consciousness have a stable metabolic rate?
//!
//! Tests whether J/Phi (Joules per unit consciousness change) converges to a
//! stable value, indicating a thermodynamic equilibrium for consciousness.
//!
//! Three conditions:
//! - FULL: thermodynamics + FEP gradient + harmony offloading (cooperation emerges)
//! - ENERGY_ONLY: thermodynamics only, no gradient/offloading (random walk under pressure)
//! - FREE: no thermodynamic enforcement (unlimited energy, control condition)
//!
//! Run: cargo run --example jphi_convergence --release
//! Output: CSV to stdout, summary to stderr

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::ConvergenceDetector;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 12;
const MAX_TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const NUM_SEEDS: usize = 10;
const CONVERGENCE_WINDOW: usize = 200;
const CONVERGENCE_THRESHOLD: f64 = 1e-3;

#[derive(Clone, Copy)]
enum Condition {
    Full,
    EnergyOnly,
    Free,
}

impl Condition {
    fn name(&self) -> &'static str {
        match self {
            Condition::Full => "FULL",
            Condition::EnergyOnly => "ENERGY_ONLY",
            Condition::Free => "FREE",
        }
    }
    fn enforce_thermo(&self) -> bool {
        !matches!(self, Condition::Free)
    }
    fn use_gradient(&self) -> bool {
        matches!(self, Condition::Full)
    }
    fn use_offloading(&self) -> bool {
        matches!(self, Condition::Full)
    }
}

struct RunResult {
    condition: &'static str,
    seed: u64,
    converged: bool,
    convergence_tick: usize,
    final_jphi: f64,
    final_phi: f64,
    alive: usize,
    final_clustering: f64,
}

fn run_experiment(condition: Condition, seed: u64) -> RunResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    if condition.enforce_thermo() {
        consciousness.constants = ThermodynamicConstants::research();
    } else {
        consciousness.constants.consciousness_maintenance_per_tick = 0.0;
        consciousness.constants.movement_cost_per_unit = 0.0;
        consciousness.constants.collision_energy_drain = 0.0;
    }

    let well_positions = vec![SVector::from([25.0, 25.0]), SVector::from([-25.0, -25.0])];
    // Well depletion: each well has finite capacity (Joules).
    // Forces migration when wells run dry, breaking equilibrium flatline.
    let mut well_remaining = vec![3000.0f64; well_positions.len()];

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 80.0;
        let y = (rng_f64(&mut rng) - 0.5) * 80.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(body) = world.body_mut(h) {
            body.linear_damping = 0.05; // LTC: tau = 20s, gentle air resistance
        }
        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );

        if let Some(entity) = consciousness.entities.get_mut(&h) {
            match i % 4 {
                0 => entity.harmony_activations = [0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.3],
                1 => entity.harmony_activations = [0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.5],
                2 => entity.harmony_activations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                _ => entity.harmony_activations = [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.3, 0.4],
            }
        }
        handles.push(h);
    }

    let mut jphi_detector = ConvergenceDetector::new(CONVERGENCE_WINDOW, CONVERGENCE_THRESHOLD);
    let mut phi_detector = ConvergenceDetector::new(CONVERGENCE_WINDOW, CONVERGENCE_THRESHOLD);
    let mut cooperation_events = 0u64;
    let mut convergence_tick = MAX_TICKS;
    let mut converged = false;

    for tick in 0..MAX_TICKS {
        // 1. Consciousness update with DYNAMIC inputs (HDC-D style)
        for &h in &handles {
            let entity = consciousness.entities.get(&h);
            let collapsed = entity.map(|e| e.energy.is_collapsed()).unwrap_or(true);
            let energy_frac = entity.map(|e| e.energy.fraction_remaining()).unwrap_or(0.0);
            let pred_error = entity.map(|e| e.prediction_error).unwrap_or(0.0);
            let motor_prec = entity.map(|e| e.motor_precision).unwrap_or(1.0);
            let harmony_total = entity.map(|e| e.total_harmony_energy()).unwrap_or(0.0);

            // Count nearby agents for this entity
            let nearby = if let Some(body) = world.body(h) {
                let pos = body.position();
                handles
                    .iter()
                    .filter(|&&oh| {
                        oh != h
                            && world
                                .body(oh)
                                .map(|ob| {
                                    (ob.position() - pos).norm()
                                        < consciousness.constants.harmony_range
                                })
                                .unwrap_or(false)
                    })
                    .count()
            } else {
                0
            };

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
                    phi: energy_frac,
                    broadcast: (nearby as f64 / 5.0).min(1.0),
                    working_memory: (1.0 - pred_error).max(0.0),
                    attention: 0.5,
                    recurrence: motor_prec,
                    embodiment: (1.0 - energy_frac * 0.3).max(0.0),
                    knowledge: (harmony_total / 8.0).min(1.0),
                    synchrony: consciousness.collective_phi.max(0.5), // Floor at 0.5 — agents in a simulation ARE synchronized
                }
            };
            let old_phi = consciousness
                .entities
                .get(&h)
                .map(|e| e.phi())
                .unwrap_or(0.0);
            consciousness.update_entity(h, &inputs, Point::origin());
            let new_phi = consciousness
                .entities
                .get(&h)
                .map(|e| e.phi())
                .unwrap_or(0.0);
            consciousness.ledger.record_phi_change(new_phi - old_phi);
        }

        // 2. FEP gradient movement (if condition uses it)
        if condition.use_gradient() {
            let agent_data: Vec<(SVector<f64, 2>, [f64; 9])> = handles
                .iter()
                .filter_map(|&h| {
                    let body = world.body(h)?;
                    let entity = consciousness.entities.get(&h)?;
                    Some((body.position(), entity.harmony_activations))
                })
                .collect();

            // Well data includes remaining fraction (depleted wells are less attractive)
            let well_data: Vec<(SVector<f64, 2>, f64)> = well_positions
                .iter()
                .zip(well_remaining.iter())
                .filter(|&(_, &rem)| rem > 0.0)
                .map(|(&p, &rem)| (p, (rem / 3000.0).min(1.0)))
                .collect();

            for &h in &handles {
                let Some(body) = world.body(h) else { continue };
                let Some(entity) = consciousness.entities.get(&h) else {
                    continue;
                };
                if entity.energy.is_collapsed() {
                    continue;
                }

                let pos = body.position();
                let ef = entity.energy.fraction_remaining();
                let harmony = entity.harmony_activations;
                let nearby: Vec<_> = agent_data
                    .iter()
                    .filter(|(p, _)| (p - pos).norm() > 2.0)
                    .cloned()
                    .collect();

                let dir = fep_gradient::free_energy_gradient(
                    &pos, ef, &harmony, &nearby, &well_data, None, 0.0,
                );
                if let Some(body) = world.body_mut(h) {
                    body.linear_velocity = dir * 25.0;
                }
            }
        }

        // 3. Maintenance + ambient regen + well regen
        let regen_mult = consciousness.resource_regeneration_multiplier();
        let maintenance_rate = consciousness.constants.consciousness_maintenance_per_tick;
        let ambient_rate = consciousness.constants.ambient_regen_rate;
        let well_rate = consciousness.constants.energy_well_regen_rate;

        // Collect which well (if any) each agent is near, with depletion check
        let agent_well_idx: Vec<Option<usize>> = handles
            .iter()
            .map(|&h| {
                world.body(h).and_then(|body| {
                    let pos = body.position();
                    well_positions
                        .iter()
                        .enumerate()
                        .find(|&(ref i, &wp)| (pos - wp).norm() < 35.0 && well_remaining[*i] > 0.0)
                        .map(|(i, _)| i)
                })
            })
            .collect();

        for (idx, &h) in handles.iter().enumerate() {
            // Energy tick reset
            if let Some(entity) = consciousness.entities.get_mut(&h) {
                entity.energy.tick_reset();
            }

            // Maintenance via ledger (records consumption + phi for J/Phi metric)
            if condition.enforce_thermo() {
                let phi = consciousness.phi(h);
                let maintenance = maintenance_rate * (1.0 + phi * 0.5);
                consciousness.consume_energy(h, maintenance);
            }

            // Regeneration (with well depletion)
            if let Some(entity) = consciousness.entities.get_mut(&h) {
                entity.energy.regenerate(ambient_rate * regen_mult);
                if let Some(wi) = agent_well_idx[idx] {
                    let draw = well_rate.min(well_remaining[wi]);
                    entity.energy.regenerate(draw);
                    well_remaining[wi] -= draw;
                }
            }
        }

        // 4. Harmony resonance + offloading
        if condition.use_offloading() {
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    let (ha, hb) = (handles[i], handles[j]);
                    let (harm_a, harm_b) = {
                        let ea = consciousness.entities.get(&ha);
                        let eb = consciousness.entities.get(&hb);
                        match (ea, eb) {
                            (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                            _ => continue,
                        }
                    };
                    let resonance = HarmonyField::<2>::resonance(&harm_a, &harm_b);
                    if resonance > 0.5 {
                        let regen = consciousness.constants.harmony_resonance_regen_rate
                            * (resonance - 0.5)
                            * 2.0;
                        if let Some(e) = consciousness.entities.get_mut(&ha) {
                            e.energy.regenerate(regen);
                        }
                        if let Some(e) = consciousness.entities.get_mut(&hb) {
                            e.energy.regenerate(regen);
                        }
                        cooperation_events += 1;
                    }
                }
            }
        }

        // 5. Physics + thermodynamics
        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        let balance = consciousness.tick_thermodynamics();

        // 6. Track convergence (normalized J/Phi per agent per second)
        if let Some(jphi) = balance.joules_per_phi {
            let alive_count = handles
                .iter()
                .filter(|h| {
                    consciousness
                        .entities
                        .get(h)
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .count()
                .max(1) as f64;
            let jphi_normalized = jphi / alive_count / (1.0 / DT); // per agent per second
            if jphi_normalized.is_finite() && jphi_detector.push(jphi_normalized) && !converged {
                converged = true;
                convergence_tick = tick;
            }
        }
        phi_detector.push(consciousness.collective_phi);

        // CSV output every 100 ticks
        if tick % 100 == 0 {
            let alive = handles
                .iter()
                .filter(|h| {
                    consciousness
                        .entities
                        .get(h)
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .count();
            let clustering = avg_nearest_neighbor(&world, &handles);
            let jphi_str = balance
                .joules_per_phi
                .map(|j| format!("{j:.6}"))
                .unwrap_or("N/A".into());
            println!(
                "{},{seed},{tick},{:.4},{jphi_str},{alive},{:.2},{clustering:.2}",
                condition.name(),
                consciousness.collective_phi,
                handles
                    .iter()
                    .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
                    .sum::<f64>()
                    / AGENTS as f64,
            );
        }
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
        .count();

    RunResult {
        condition: condition.name(),
        seed,
        converged,
        convergence_tick,
        final_jphi: jphi_detector.rolling_mean(),
        final_phi: phi_detector.rolling_mean(),
        alive,
        final_clustering: avg_nearest_neighbor(&world, &handles),
    }
}

fn main() {
    eprintln!("=== J/Phi Convergence Experiment ===");
    eprintln!("Agents: {AGENTS}, MaxTicks: {MAX_TICKS}, Seeds: {NUM_SEEDS}");
    eprintln!("Window: {CONVERGENCE_WINDOW}, Threshold: {CONVERGENCE_THRESHOLD}");

    // CSV header
    println!("condition,seed,tick,collective_phi,joules_per_phi,alive,avg_energy,clustering");

    let conditions = [Condition::Full, Condition::EnergyOnly, Condition::Free];

    for condition in &conditions {
        let mut results = Vec::new();
        for s in 0..NUM_SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  Running {} seed={seed}...", condition.name());
            results.push(run_experiment(*condition, seed));
        }

        // Summary
        let n = results.len() as f64;
        let converged_count = results.iter().filter(|r| r.converged).count();
        let mean_tick = if converged_count > 0 {
            results
                .iter()
                .filter(|r| r.converged)
                .map(|r| r.convergence_tick as f64)
                .sum::<f64>()
                / converged_count as f64
        } else {
            f64::NAN
        };
        let mean_jphi = results.iter().map(|r| r.final_jphi).sum::<f64>() / n;
        let mean_phi = results.iter().map(|r| r.final_phi).sum::<f64>() / n;
        let mean_alive = results.iter().map(|r| r.alive as f64).sum::<f64>() / n;
        let mean_cluster = results.iter().map(|r| r.final_clustering).sum::<f64>() / n;

        let jphi_vals: Vec<f64> = results.iter().map(|r| r.final_jphi).collect();
        let jphi_std = std_dev(&jphi_vals);
        let jphi_ci = 1.96 * jphi_std / n.sqrt();

        eprintln!(
            "\n── {} ({}/{} converged) ──",
            condition.name(),
            converged_count,
            NUM_SEEDS
        );
        eprintln!("  Convergence tick:  {:.0} (mean of converged)", mean_tick);
        eprintln!(
            "  J/Φ:              {:.6} ± {:.6} (95% CI)",
            mean_jphi, jphi_ci
        );
        eprintln!("  Final Φ:          {:.4}", mean_phi);
        eprintln!("  Alive:            {:.1}/{AGENTS}", mean_alive);
        eprintln!("  Clustering:       {:.2}", mean_cluster);
    }

    // Statistical comparison: FULL vs FREE clustering
    eprintln!("\n── Statistical Tests ──");

    // Re-run to collect per-condition data for comparison
    let mut full_cluster = Vec::new();
    let mut free_cluster = Vec::new();
    let mut energy_cluster = Vec::new();

    for s in 0..NUM_SEEDS {
        let seed = 42 + s as u64 * 997;
        let r_full = run_experiment(Condition::Full, seed);
        let r_free = run_experiment(Condition::Free, seed);
        let r_energy = run_experiment(Condition::EnergyOnly, seed);
        full_cluster.push(r_full.final_clustering);
        free_cluster.push(r_free.final_clustering);
        energy_cluster.push(r_energy.final_clustering);
    }

    use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};

    let (u, z, p) = mann_whitney_u(&full_cluster, &free_cluster);
    let d = cohens_d(&full_cluster, &free_cluster);
    eprintln!("  FULL vs FREE clustering:");
    eprintln!("    Mann-Whitney U={:.1}, z={:.3}, p={:.4}", u, z, p);
    eprintln!("    Cohen's d={:.3} ({})", d, effect_label(d));
    eprintln!(
        "    {} at α=0.05",
        if p < 0.05 {
            "SIGNIFICANT"
        } else {
            "not significant"
        }
    );

    let (u2, z2, p2) = mann_whitney_u(&full_cluster, &energy_cluster);
    let d2 = cohens_d(&full_cluster, &energy_cluster);
    eprintln!("  FULL vs ENERGY_ONLY clustering:");
    eprintln!("    Mann-Whitney U={:.1}, z={:.3}, p={:.4}", u2, z2, p2);
    eprintln!("    Cohen's d={:.3} ({})", d2, effect_label(d2));
    eprintln!(
        "    {} at α=0.05",
        if p2 < 0.05 {
            "SIGNIFICANT"
        } else {
            "not significant"
        }
    );

    eprintln!("\n=== Experiment Complete ===");
}

fn effect_label(d: f64) -> &'static str {
    let d = d.abs();
    if d < 0.2 {
        "negligible"
    } else if d < 0.5 {
        "small"
    } else if d < 0.8 {
        "medium"
    } else {
        "large"
    }
}

fn avg_nearest_neighbor(world: &PhysicsWorld<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let positions: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position()))
        .collect();
    if positions.len() < 2 {
        return f64::MAX;
    }
    let mut total = 0.0;
    for (i, p) in positions.iter().enumerate() {
        let nearest = positions
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, q)| (p - q).norm())
            .fold(f64::MAX, f64::min);
        total += nearest;
    }
    total / positions.len() as f64
}

fn std_dev(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / n;
    (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn rng_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
