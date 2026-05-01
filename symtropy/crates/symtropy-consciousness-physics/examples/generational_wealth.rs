// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generational Wealth — does position inheritance create dynasties?
//!
//! Combines evolution (dead agents replaced by offspring) with inequality
//! tracking. Offspring spawn near their parent and inherit the parent's
//! harmony profile (with mutation). Tests whether "being born near a well"
//! creates persistent advantage across generations.
//!
//! Run: cargo run --example generational_wealth --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::cohens_d;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 20_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 6;
const REPRODUCTION_INTERVAL: usize = 1000;
const MUTATION_RATE: f64 = 0.05;

struct WealthResult {
    generations: f64,
    final_alive: f64,
    gini_early: f64,       // gen 1-3
    gini_late: f64,        // final gen
    dynasty_fraction: f64, // fraction of final population descended from initial top 25%
    mean_resonance: f64,
    lineage_diversity: f64, // unique parent lineages in final pop
}

fn run_experiment(seed: u64) -> WealthResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    consciousness.constants = constants;

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![6000.0f64; 2];

    let mut rng = seed;
    let mut handles = Vec::new();
    let mut agent_harmonies: Vec<[f64; 9]> = Vec::new();
    let mut agent_lineage: Vec<usize> = Vec::new(); // tracks which original agent they descend from
    let mut generation = 0usize;

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
        let harmony: [f64; 9] = std::array::from_fn(|_| rng_f64(&mut rng) * 0.7 + 0.15);
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = harmony;
        }
        handles.push(h);
        agent_harmonies.push(harmony);
        agent_lineage.push(i); // each agent is their own lineage founder
    }

    // Record which agents are initially in top 25% by well proximity
    let initial_well_proximity: Vec<f64> = handles
        .iter()
        .map(|&h| {
            world
                .body(h)
                .map(|b| {
                    let pos = b.position().0;
                    wells
                        .iter()
                        .map(|w| (pos - w).norm())
                        .fold(f64::MAX, f64::min)
                })
                .unwrap_or(f64::MAX)
        })
        .collect();
    let mut sorted_prox: Vec<(usize, f64)> = initial_well_proximity
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d))
        .collect();
    sorted_prox.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // closest first
    let top_founders: Vec<usize> = sorted_prox[..AGENTS / 4].iter().map(|(i, _)| *i).collect();

    let mut gini_samples = Vec::new();

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
            .map(|(&p, &r)| (p, (r / 6000.0).min(1.0)))
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

        // Gini sampling
        if tick % 2000 == 0 {
            let energies: Vec<f64> = handles
                .iter()
                .map(|h| {
                    consciousness
                        .entities
                        .get(h)
                        .map(|e| e.energy.available)
                        .unwrap_or(0.0)
                })
                .collect();
            gini_samples.push(gini(&energies));
        }

        // Evolution
        if tick > 0 && tick % REPRODUCTION_INTERVAL == 0 {
            generation += 1;
            let alive_idx: Vec<usize> = (0..handles.len())
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&handles[i])
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .collect();
            let dead_idx: Vec<usize> = (0..handles.len())
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&handles[i])
                        .map(|e| e.energy.is_collapsed())
                        .unwrap_or(true)
                })
                .collect();

            if !alive_idx.is_empty() && !dead_idx.is_empty() {
                for &di in &dead_idx {
                    // Fittest parent = most energy
                    let parent = *alive_idx
                        .iter()
                        .max_by(|&&a, &&b| {
                            let ea = consciousness
                                .entities
                                .get(&handles[a])
                                .map(|e| e.energy.available)
                                .unwrap_or(0.0);
                            let eb = consciousness
                                .entities
                                .get(&handles[b])
                                .map(|e| e.energy.available)
                                .unwrap_or(0.0);
                            ea.partial_cmp(&eb).unwrap()
                        })
                        .unwrap();

                    let parent_harmony = agent_harmonies[parent];
                    let mut child_harmony = parent_harmony;
                    for k in 0..8 {
                        child_harmony[k] += (rng_f64(&mut rng) - 0.5) * MUTATION_RATE * 2.0;
                        child_harmony[k] = child_harmony[k].clamp(0.0, 1.0);
                    }

                    let h = handles[di];
                    if let Some(e) = consciousness.entities.get_mut(&h) {
                        e.energy.available = consciousness.constants.initial_energy;
                        e.energy.collapsed = false;
                        e.harmony_activations = child_harmony;
                        e.prediction_error = 0.0;
                    }
                    if let Some(parent_body) = world.body(handles[parent]) {
                        let parent_pos = parent_body.position().0;
                        if let Some(b) = world.body_mut(h) {
                            let offset = SVector::from([
                                (rng_f64(&mut rng) - 0.5) * 10.0,
                                (rng_f64(&mut rng) - 0.5) * 10.0,
                            ]);
                            b.transform.translation = Point(parent_pos + offset);
                            b.linear_velocity = SVector::zeros();
                        }
                    }
                    agent_harmonies[di] = child_harmony;
                    agent_lineage[di] = agent_lineage[parent]; // inherit lineage
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    let final_alive = handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count() as f64;

    // Dynasty fraction: what % of final population descends from initial top-25% (well-adjacent)?
    let dynasty_count = agent_lineage
        .iter()
        .filter(|&&l| top_founders.contains(&l))
        .count() as f64;
    let dynasty_fraction = dynasty_count / AGENTS as f64;

    // Lineage diversity
    let mut unique_lineages: Vec<usize> = agent_lineage.clone();
    unique_lineages.sort();
    unique_lineages.dedup();
    let lineage_diversity = unique_lineages.len() as f64;

    // Mean resonance
    let harms: Vec<[f64; 9]> = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.harmony_activations))
        .collect();
    let mut res_total = 0.0;
    let mut res_count = 0;
    for i in 0..harms.len() {
        for j in (i + 1)..harms.len() {
            res_total += HarmonyField::<2>::resonance(&harms[i], &harms[j]);
            res_count += 1;
        }
    }
    let mean_resonance = if res_count > 0 {
        res_total / res_count as f64
    } else {
        0.0
    };

    let gini_early = if gini_samples.len() >= 3 {
        (gini_samples[1] + gini_samples[2]) / 2.0
    } else {
        0.0
    };
    let gini_late = *gini_samples.last().unwrap_or(&0.0);

    WealthResult {
        generations: generation as f64,
        final_alive,
        gini_early,
        gini_late,
        dynasty_fraction,
        mean_resonance,
        lineage_diversity,
    }
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

fn main() {
    eprintln!("=== Generational Wealth Experiment ===");
    eprintln!("Does position inheritance create dynasties?");
    eprintln!(
        "{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds, reproduction every {REPRODUCTION_INTERVAL}"
    );

    println!("seed,generations,alive,gini_early,gini_late,dynasty_frac,mean_res,lineage_diversity");

    let mut results = Vec::new();
    for s in 0..SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  seed={seed}...");
        let r = run_experiment(seed);
        println!(
            "{seed},{:.0},{:.1},{:.4},{:.4},{:.3},{:.4},{:.0}",
            r.generations,
            r.final_alive,
            r.gini_early,
            r.gini_late,
            r.dynasty_fraction,
            r.mean_resonance,
            r.lineage_diversity
        );
        results.push(r);
    }

    let n = results.len() as f64;
    eprintln!("\n── Generational Wealth ──");
    eprintln!(
        "  Generations:    {:.0}",
        results.iter().map(|r| r.generations).sum::<f64>() / n
    );
    eprintln!(
        "  Final alive:    {:.1}/{AGENTS}",
        results.iter().map(|r| r.final_alive).sum::<f64>() / n
    );
    eprintln!(
        "  Gini (early):   {:.3}",
        results.iter().map(|r| r.gini_early).sum::<f64>() / n
    );
    eprintln!(
        "  Gini (late):    {:.3}",
        results.iter().map(|r| r.gini_late).sum::<f64>() / n
    );
    eprintln!(
        "  Dynasty frac:   {:.1}% (descendants of initial top-25%)",
        results.iter().map(|r| r.dynasty_fraction).sum::<f64>() / n * 100.0
    );
    eprintln!(
        "  Lineage diversity: {:.1} unique lineages (of {AGENTS} original)",
        results.iter().map(|r| r.lineage_diversity).sum::<f64>() / n
    );
    eprintln!(
        "  Mean resonance: {:.4}",
        results.iter().map(|r| r.mean_resonance).sum::<f64>() / n
    );

    let expected_dynasty = 25.0; // if random, 25% would descend from top-25%
    let actual = results.iter().map(|r| r.dynasty_fraction).sum::<f64>() / n * 100.0;
    eprintln!("\n── Dynasty Effect ──");
    if actual > expected_dynasty + 10.0 {
        eprintln!("  DYNASTY: {actual:.1}% descend from well-adjacent founders (expected 25%)");
        eprintln!("  Position privilege persists across generations");
    } else if actual < expected_dynasty - 10.0 {
        eprintln!("  ANTI-DYNASTY: {actual:.1}% from well-adjacent (expected 25%)");
        eprintln!("  Starting advantage does NOT persist");
    } else {
        eprintln!("  NEUTRAL: {actual:.1}% from well-adjacent (expected 25%)");
        eprintln!("  No dynasty effect detected");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
