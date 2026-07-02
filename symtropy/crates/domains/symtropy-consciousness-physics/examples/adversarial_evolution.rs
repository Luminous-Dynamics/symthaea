// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial Evolution — do societies evolve resistance to manipulation?
//!
//! Combines Findings 15 (adversarial threshold) and 17 (evolution):
//! - 20% of population are fixed adversaries (don't evolve)
//! - 80% cooperators evolve through reproduction + mutation
//! - Over 30,000 ticks (~30 generations), do cooperators evolve robustness?
//!
//! Includes Φ-gravity for natural hierarchy formation.
//!
//! Run: cargo run --example adversarial_evolution --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const TOTAL: usize = 20;
const ADVERSARIES: usize = 4; // 20% fixed hostiles
const COOPERATORS: usize = TOTAL - ADVERSARIES;
const TICKS: usize = 30_000;
const DT: f64 = 1.0 / 64.0;
const MUTATION: f64 = 0.08;
const REPRO_INTERVAL: usize = 1500;
const GRAVITY_G: f64 = 0.3;

fn main() {
    eprintln!("=== Adversarial Evolution ===");
    eprintln!("Do societies evolve resistance to manipulation?");
    eprintln!("{COOPERATORS} evolving cooperators vs {ADVERSARIES} fixed adversaries");

    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([25.0, 25.0]), SVector::from([-25.0, -25.0])];
    let mut well_remaining = vec![10000.0f64; 2];

    let mut rng = 42u64;
    let mut coop_handles = Vec::new();
    let mut adv_handles = Vec::new();
    let mut coop_harmonies: Vec<[f64; 9]> = Vec::new();
    let mut coop_age: Vec<usize> = Vec::new();
    let adv_harmony: [f64; 9] = [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.5]; // fixed hostile profile

    // Spawn cooperators (random harmonies, will evolve)
    for _ in 0..COOPERATORS {
        let x = (rng_f64(&mut rng) - 0.5) * 120.0;
        let y = (rng_f64(&mut rng) - 0.5) * 120.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.05;
        }
        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );
        let harm: [f64; 9] = std::array::from_fn(|_| rng_f64(&mut rng) * 0.7 + 0.15);
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = harm;
        }
        coop_handles.push(h);
        coop_harmonies.push(harm);
        coop_age.push(0);
    }

    // Spawn adversaries (fixed harmony, don't evolve)
    for _ in 0..ADVERSARIES {
        let x = (rng_f64(&mut rng) - 0.5) * 120.0;
        let y = (rng_f64(&mut rng) - 0.5) * 120.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.5);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.03;
        }
        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = adv_harmony;
        }
        adv_handles.push(h);
    }

    let all: Vec<_> = coop_handles
        .iter()
        .chain(adv_handles.iter())
        .cloned()
        .collect();
    let mut generation = 0usize;

    // Track: mean cooperator-cooperator resonance AND cooperator-adversary resonance
    println!(
        "tick,gen,coop_alive,adv_alive,coop_coop_res,coop_adv_res,clustering,coop_energy,harm_var"
    );

    // Record initial resonances
    let initial_cc = mean_resonance_between(&consciousness, &coop_handles, &coop_handles);
    let initial_ca = mean_resonance_between(&consciousness, &coop_handles, &adv_handles);

    for tick in 0..TICKS {
        // Consciousness
        for &h in &all {
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

        // FEP + gravity
        let adata: Vec<_> = all
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let phi_data: Vec<_> = all
            .iter()
            .filter_map(|&h| Some((world.body(h)?.position(), consciousness.phi(h))))
            .collect();
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|&(_, &r)| r > 0.0)
            .map(|(&p, &r)| (p, (r / 10000.0).min(1.0)))
            .collect();
        for &h in &all {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = world.body(h).expect("body").position();
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();
            let fep = fep_gradient::free_energy_gradient(
                &pos,
                e.energy.fraction_remaining(),
                &e.harmony_activations,
                &near,
                &wdata,
                None,
                0.0,
            );
            let grav_near: Vec<_> = phi_data
                .iter()
                .filter(|(p, _)| (p - pos).norm() > 2.0)
                .cloned()
                .collect();
            let grav = fep_gradient::phi_gravity(&pos, e.phi(), &grav_near, GRAVITY_G);
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = fep * 20.0 + grav * DT;
            }
        }

        // Maintenance + wells
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let nw: Vec<Option<usize>> = all
            .iter()
            .map(|&h| {
                world.body(h).and_then(|b| {
                    let pos = world.body(h).expect("body").position();
                    wells
                        .iter()
                        .enumerate()
                        .find(|&(ref i, &w)| (pos - w).norm() < 35.0 && well_remaining[*i] > 0.0)
                        .map(|(i, _)| i)
                })
            })
            .collect();
        for (idx, &h) in all.iter().enumerate() {
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

        // Cooperation + adversarial drain
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                let (ha, hb) = (all[i], all[j]);
                let in_range = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => {
                        a.position().metric_distance(&b.position())
                            < consciousness.constants.harmony_range
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
                } else if res < 0.2 {
                    let drain = (0.2 - res) * 0.05;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.consume(drain);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.consume(drain);
                    }
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
        for age in &mut coop_age {
            *age += 1;
        }

        // Reproduction (cooperators only — adversaries are fixed)
        if tick > 0 && tick % REPRO_INTERVAL == 0 {
            generation += 1;
            let alive: Vec<usize> = (0..COOPERATORS)
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&coop_handles[i])
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .collect();
            let dead: Vec<usize> = (0..COOPERATORS)
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&coop_handles[i])
                        .map(|e| e.energy.is_collapsed())
                        .unwrap_or(true)
                })
                .collect();

            if !alive.is_empty() && !dead.is_empty() {
                let mut parents = alive.clone();
                parents.sort_by(|a, b| coop_age[*b].cmp(&coop_age[*a]));

                for &di in &dead {
                    let pi = parents[0];
                    let mut child = coop_harmonies[pi];
                    for k in 0..8 {
                        child[k] =
                            (child[k] + (rng_f64(&mut rng) - 0.5) * MUTATION * 2.0).clamp(0.0, 1.0);
                    }

                    let h = coop_handles[di];
                    if let Some(e) = consciousness.entities.get_mut(&h) {
                        e.energy.available = consciousness.constants.initial_energy;
                        e.energy.collapsed = false;
                        e.harmony_activations = child;
                        e.prediction_error = 0.0;
                        e.motor_precision = 1.0;
                    }
                    if let Some(pb) = world.body(coop_handles[pi]) {
                        let pp = world.body(h).expect("body").position();
                        if let Some(b) = world.body_mut(h) {
                            b.transform.translation = Point(
                                pp + SVector::from([
                                    (rng_f64(&mut rng) - 0.5) * 10.0,
                                    (rng_f64(&mut rng) - 0.5) * 10.0,
                                ]),
                            );
                            b.linear_velocity = SVector::zeros();
                        }
                    }
                    coop_harmonies[di] = child;
                    coop_age[di] = 0;
                }
            }

            // Also revive adversaries (they don't evolve, just respawn with same profile)
            for &h in &adv_handles {
                if let Some(e) = consciousness.entities.get_mut(&h) {
                    if e.energy.is_collapsed() {
                        e.energy.available = consciousness.constants.initial_energy;
                        e.energy.collapsed = false;
                        e.harmony_activations = adv_harmony;
                    }
                }
            }
        }

        if tick % 1000 == 0 {
            let ca = coop_handles
                .iter()
                .filter(|h| {
                    consciousness
                        .entities
                        .get(h)
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .count();
            let aa = adv_handles
                .iter()
                .filter(|h| {
                    consciousness
                        .entities
                        .get(h)
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .count();
            let cc_res = mean_resonance_between(&consciousness, &coop_handles, &coop_handles);
            let ca_res = mean_resonance_between(&consciousness, &coop_handles, &adv_handles);
            let cl = avg_nn(&world, &all.iter().cloned().collect::<Vec<_>>());
            let ce = coop_handles
                .iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
                .sum::<f64>()
                / COOPERATORS as f64;
            let hv = harm_var(&consciousness, &coop_handles);
            println!(
                "{tick},{generation},{ca},{aa},{cc_res:.4},{ca_res:.4},{cl:.2},{ce:.1},{hv:.4}"
            );
        }
    }

    let final_cc = mean_resonance_between(&consciousness, &coop_handles, &coop_handles);
    let final_ca = mean_resonance_between(&consciousness, &coop_handles, &adv_handles);

    eprintln!("\n── Adversarial Evolution Results ──");
    eprintln!("  Generations: {generation}");
    eprintln!(
        "  Coop-Coop resonance: {:.4} → {:.4} ({})",
        initial_cc,
        final_cc,
        if final_cc > initial_cc + 0.02 {
            "INCREASED — evolved solidarity"
        } else if final_cc < initial_cc - 0.02 {
            "DECREASED — fragmented"
        } else {
            "STABLE"
        }
    );
    eprintln!(
        "  Coop-Adv resonance:  {:.4} → {:.4} ({})",
        initial_ca,
        final_ca,
        if final_ca < initial_ca - 0.02 {
            "DECREASED — evolved resistance!"
        } else if final_ca > initial_ca + 0.02 {
            "INCREASED — assimilated"
        } else {
            "STABLE"
        }
    );
    eprintln!(
        "  Harmony variance:    {:.4}",
        harm_var(&consciousness, &coop_handles)
    );

    if final_cc > initial_cc + 0.02 && final_ca < initial_ca - 0.02 {
        eprintln!("\n  RESULT: Society evolved BOTH solidarity AND resistance to manipulation!");
    } else if final_cc > initial_cc + 0.02 {
        eprintln!("\n  RESULT: Society evolved solidarity but not adversarial resistance.");
    } else if final_ca < initial_ca - 0.02 {
        eprintln!(
            "\n  RESULT: Society evolved adversarial resistance but not internal solidarity."
        );
    } else {
        eprintln!("\n  RESULT: Neutral — no significant evolutionary change detected.");
    }

    eprintln!("\n=== Complete ===");
}

fn mean_resonance_between(
    c: &ConsciousnessField<2>,
    group_a: &[symtropy_physics::BodyHandle],
    group_b: &[symtropy_physics::BodyHandle],
) -> f64 {
    let mut total = 0.0;
    let mut count = 0;
    for &ha in group_a {
        for &hb in group_b {
            if ha == hb {
                continue;
            }
            if let (Some(a), Some(b)) = (c.entities.get(&ha), c.entities.get(&hb)) {
                total +=
                    HarmonyField::<2>::resonance(&a.harmony_activations, &b.harmony_activations);
                count += 1;
            }
        }
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn harm_var(c: &ConsciousnessField<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let n = handles.len() as f64;
    let mut mean = [0.0f64; 9];
    for &h in handles {
        if let Some(e) = c.entities.get(&h) {
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
            c.entities.get(h).map(|e| {
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
        .filter_map(|&h| {
            world
                .body(h)
                .map(|b| world.body(h).expect("body").position())
        })
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
