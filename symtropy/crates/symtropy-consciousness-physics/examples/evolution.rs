// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evolutionary Selection — does cooperation survive natural selection?
//!
//! Agents that survive longest reproduce (offspring inherit parent's harmony
//! profile + small mutation). Dead agents are replaced by offspring.
//! Tests whether thermodynamic pressure selects for cooperative harmony
//! profiles or selfish ones.
//!
//! Includes Φ-gravity: high-integration agents gravitationally attract others.
//!
//! Run: cargo run --example evolution --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 20_000;
const DT: f64 = 1.0 / 64.0;
const MUTATION_RATE: f64 = 0.05; // per-harmony-channel mutation magnitude
const REPRODUCTION_INTERVAL: usize = 1000; // ticks between reproduction events
const GRAVITY_CONSTANT: f64 = 0.3; // Φ-gravity coupling

fn main() {
    eprintln!("=== Evolutionary Selection Experiment ===");
    eprintln!("Does cooperation survive natural selection?");
    eprintln!("Agents: {AGENTS}, Ticks: {TICKS}, Reproduction every {REPRODUCTION_INTERVAL} ticks");
    eprintln!("Φ-gravity: G={GRAVITY_CONSTANT}, Mutation: {MUTATION_RATE}");

    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([30.0, 30.0]), SVector::from([-30.0, -30.0])];
    let mut well_remaining = vec![8000.0f64; 2]; // Large wells for long runs

    let mut rng = 42u64;
    let mut handles = Vec::new();
    let mut agent_harmonies: Vec<[f64; 9]> = Vec::new();
    let mut agent_age: Vec<usize> = Vec::new();
    let mut generation = 0usize;

    // Initial population: random harmony profiles
    for _ in 0..AGENTS {
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

        let harmony: [f64; 9] = std::array::from_fn(|_| rng_f64(&mut rng) * 0.8 + 0.1);
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = harmony;
        }
        handles.push(h);
        agent_harmonies.push(harmony);
        agent_age.push(0);
    }

    // Track evolutionary metrics over time
    println!(
        "tick,generation,alive,mean_resonance,mean_phi,clustering,mean_energy,harmony_variance"
    );

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

        // FEP gradient + Φ-gravity
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position().0,
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let phi_data: Vec<_> = handles
            .iter()
            .filter_map(|&h| Some((world.body(h)?.position().0, consciousness.phi(h))))
            .collect();
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|(_, &r)| r > 0.0)
            .map(|(&p, &r)| (p, (r / 8000.0).min(1.0)))
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
            let self_phi = e.phi();
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    (p - pos).norm() > 2.0
                        && (p - pos).norm() < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();

            // FEP gradient (behavioral)
            let fep_dir = fep_gradient::free_energy_gradient(
                &pos,
                e.energy.fraction_remaining(),
                &e.harmony_activations,
                &near,
                &wdata,
                None,
                0.0,
            );

            // Φ-gravity (gravitational attraction to high-Φ agents)
            let gravity_agents: Vec<_> = phi_data
                .iter()
                .filter(|(p, _)| (p - pos).norm() > 2.0)
                .cloned()
                .collect();
            let gravity =
                fep_gradient::phi_gravity(&pos, self_phi, &gravity_agents, GRAVITY_CONSTANT);

            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = fep_dir * 20.0 + gravity * DT;
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

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();

        // Age all agents
        for age in &mut agent_age {
            *age += 1;
        }

        // EVOLUTION: every REPRODUCTION_INTERVAL ticks, replace dead agents with offspring
        if tick > 0 && tick % REPRODUCTION_INTERVAL == 0 {
            generation += 1;

            // Find dead and alive agents
            let alive_indices: Vec<usize> = (0..handles.len())
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&handles[i])
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .collect();
            let dead_indices: Vec<usize> = (0..handles.len())
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&handles[i])
                        .map(|e| e.energy.is_collapsed())
                        .unwrap_or(true)
                })
                .collect();

            if !alive_indices.is_empty() && !dead_indices.is_empty() {
                // Sort alive by age (oldest = most fit = most likely to reproduce)
                let mut parents: Vec<usize> = alive_indices.clone();
                parents.sort_by(|a, b| agent_age[*b].cmp(&agent_age[*a]));

                for &dead_idx in &dead_indices {
                    // Pick a parent (weighted toward oldest survivors)
                    let parent_idx = parents[0]; // most fit

                    // Mutate parent's harmony
                    let parent_harmony = agent_harmonies[parent_idx];
                    let mut child_harmony = parent_harmony;
                    for k in 0..8 {
                        child_harmony[k] += (rng_f64(&mut rng) - 0.5) * MUTATION_RATE * 2.0;
                        child_harmony[k] = child_harmony[k].clamp(0.0, 1.0);
                    }

                    // Respawn dead agent as offspring
                    let h = handles[dead_idx];
                    if let Some(e) = consciousness.entities.get_mut(&h) {
                        e.energy.available = consciousness.constants.initial_energy;
                        e.energy.collapsed = false;
                        e.harmony_activations = child_harmony;
                        e.prediction_error = 0.0;
                        e.motor_precision = 1.0;
                    }

                    // Reposition near parent
                    if let Some(parent_body) = world.body(handles[parent_idx]) {
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

                    agent_harmonies[dead_idx] = child_harmony;
                    agent_age[dead_idx] = 0;
                }

                eprintln!(
                    "  Gen {generation} (tick {tick}): {}/{AGENTS} alive, {} reproduced",
                    alive_indices.len(),
                    dead_indices.len()
                );
            }
        }

        // Output every 500 ticks
        if tick % 500 == 0 {
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
            let mean_res = compute_mean_resonance(&consciousness, &handles);
            let mean_phi = consciousness.collective_phi;
            let clustering = avg_nn(&world, &handles);
            let mean_energy = handles
                .iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
                .sum::<f64>()
                / AGENTS as f64;
            let harm_var = compute_harmony_variance(&consciousness, &handles);

            println!("{tick},{generation},{alive},{mean_res:.4},{mean_phi:.4},{clustering:.2},{mean_energy:.1},{harm_var:.4}");
        }
    }

    // Final analysis
    eprintln!("\n── Evolutionary Outcome ──");
    let final_res = compute_mean_resonance(&consciousness, &handles);
    let initial_expected_res = 0.5; // random harmonies ≈ 0.5 mean resonance
    eprintln!(
        "  Final mean resonance: {:.4} (started ~{:.1})",
        final_res, initial_expected_res
    );
    eprintln!("  Generations: {generation}");
    eprintln!(
        "  Final harmony variance: {:.4}",
        compute_harmony_variance(&consciousness, &handles)
    );

    if final_res > initial_expected_res + 0.05 {
        eprintln!("  RESULT: Cooperation EVOLVED — resonance increased through selection");
    } else if final_res < initial_expected_res - 0.05 {
        eprintln!("  RESULT: Selfishness evolved — resonance decreased");
    } else {
        eprintln!("  RESULT: Neutral — resonance unchanged by selection");
    }

    eprintln!("\n=== Complete ===");
}

fn compute_mean_resonance(
    consciousness: &ConsciousnessField<2>,
    handles: &[symtropy_physics::BodyHandle],
) -> f64 {
    let harms: Vec<[f64; 9]> = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.harmony_activations))
        .collect();
    if harms.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0;
    for i in 0..harms.len() {
        for j in (i + 1)..harms.len() {
            total += HarmonyField::<2>::resonance(&harms[i], &harms[j]);
            count += 1;
        }
    }
    if count > 0 {
        total / count as f64
    } else {
        0.0
    }
}

fn compute_harmony_variance(
    consciousness: &ConsciousnessField<2>,
    handles: &[symtropy_physics::BodyHandle],
) -> f64 {
    let n = handles.len() as f64;
    let mut mean = [0.0f64; 9];
    for &h in handles {
        if let Some(e) = consciousness.entities.get(&h) {
            for i in 0..8 {
                mean[i] += e.harmony_activations[i];
            }
        }
    }
    for v in &mut mean {
        *v /= n;
    }
    handles
        .iter()
        .filter_map(|h| {
            consciousness.entities.get(h).map(|e| {
                (0..8)
                    .map(|i| (e.harmony_activations[i] - mean[i]).powi(2))
                    .sum::<f64>()
            })
        })
        .sum::<f64>()
        / n
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
