// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial Learning — do cooperators need adaptation when adversaries evolve?
//!
//! From Finding 18: cooperators evolve solidarity against FIXED adversaries.
//! But what if adversaries also learn? This pits reward-modulated plasticity
//! cooperators against reward-modulated plasticity adversaries.
//!
//! 3 conditions:
//! 1. FIXED_VS_FIXED: Neither side learns (baseline from F14-18)
//! 2. LEARN_VS_FIXED: Cooperators learn, adversaries fixed
//! 3. LEARN_VS_LEARN: Both sides learn (arms race)
//!
//! The hypothesis: learning is unnecessary against fixed adversaries (F31),
//! but REQUIRED against adaptive adversaries.
//!
//! Run: cargo run --example adversarial_learning --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient::{self, LearnedFepWeights};
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const TOTAL: usize = 20;
const ADVERSARIES: usize = 5; // 25% (critical threshold from F14)
const COOPERATORS: usize = TOTAL - ADVERSARIES;
const TICKS: usize = 12_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;
const EVOLUTION_INTERVAL: usize = 1000;
const MUTATION_RATE: f64 = 0.05;

#[derive(Clone, Copy)]
enum Mode {
    FixedFixed,
    LearnFixed,
    LearnLearn,
}

struct AdvLearnResult {
    condition: &'static str,
    coop_alive: f64,
    adv_alive: f64,
    coop_energy: f64,
    adv_energy: f64,
    coop_weight_drift: f64,
    adv_weight_drift: f64,
    cooperation_events: f64,
    generations: f64,
}

fn run_experiment(mode: Mode, seed: u64) -> AdvLearnResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    consciousness.constants = constants;

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![3000.0f64; 2];

    let mut rng = seed;
    let mut coop_handles = Vec::new();
    let mut adv_handles = Vec::new();
    let mut coop_weights: Vec<LearnedFepWeights> = Vec::new();
    let mut adv_weights: Vec<LearnedFepWeights> = Vec::new();
    let mut coop_harmonies: Vec<[f64; 9]> = Vec::new();
    let mut adv_harmonies: Vec<[f64; 9]> = Vec::new();

    let coop_learn = matches!(mode, Mode::LearnFixed | Mode::LearnLearn);
    let adv_learn = matches!(mode, Mode::LearnLearn);

    // Spawn cooperators
    for i in 0..COOPERATORS {
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
        let harmony = match i % 3 {
            0 => [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
            1 => [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
            _ => [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
        };
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = harmony;
        }
        coop_handles.push(h);
        coop_weights.push(LearnedFepWeights::default());
        coop_harmonies.push(harmony);
    }

    // Spawn adversaries (inverted harmony = anti-resonant)
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
        let harmony = [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.5];
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = harmony;
        }
        adv_handles.push(h);
        adv_weights.push(LearnedFepWeights::default());
        adv_harmonies.push(harmony);
    }

    let all_handles: Vec<_> = coop_handles
        .iter()
        .chain(adv_handles.iter())
        .cloned()
        .collect();
    let mut coop_events = 0u64;
    let mut generation = 0usize;

    let cond_name = match mode {
        Mode::FixedFixed => "FIXED_v_FIXED",
        Mode::LearnFixed => "LEARN_v_FIXED",
        Mode::LearnLearn => "LEARN_v_LEARN",
    };

    for tick in 0..TICKS {
        // Consciousness update
        for &h in &all_handles {
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

        // Record energy for memory
        for (idx, &h) in coop_handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.memory.record_energy(e.energy.fraction_remaining());
            }
        }
        for (idx, &h) in adv_handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.memory.record_energy(e.energy.fraction_remaining());
            }
        }

        // FEP gradient with optional learning
        let adata: Vec<_> = all_handles
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
            .map(|(&p, &r)| (p, (r / 3000.0).min(1.0)))
            .collect();

        // Cooperator movement
        for (idx, &h) in coop_handles.iter().enumerate() {
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
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();

            if coop_learn {
                let (dir, contribs) = fep_gradient::free_energy_gradient_learned(
                    &pos,
                    e.energy.fraction_remaining(),
                    None,
                    &e.harmony_activations,
                    &near,
                    &wdata,
                    None,
                    0.0,
                    &coop_weights[idx],
                );
                let reward = e.memory.windowed_reward();
                coop_weights[idx].update_windowed(reward, contribs);
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

        // Adversary movement
        for (idx, &h) in adv_handles.iter().enumerate() {
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
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();

            if adv_learn {
                let (dir, contribs) = fep_gradient::free_energy_gradient_learned(
                    &pos,
                    e.energy.fraction_remaining(),
                    None,
                    &e.harmony_activations,
                    &near,
                    &wdata,
                    None,
                    0.0,
                    &adv_weights[idx],
                );
                let reward = e.memory.windowed_reward();
                adv_weights[idx].update_windowed(reward, contribs);
                if let Some(b) = world.body_mut(h) {
                    b.linear_velocity = dir * 25.0;
                } // adversaries faster
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
                    b.linear_velocity = dir * 25.0;
                }
            }
        }

        // Maintenance
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        for &h in all_handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                if let Some(b) = world.body(h) {
                    let pos = b.position();
                    for (wi, &w) in wells.iter().enumerate() {
                        if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                            let d = wr.min(well_remaining[wi]);
                            e.energy.regenerate(d);
                            well_remaining[wi] -= d;
                            break;
                        }
                    }
                }
            }
        }

        // Cooperation + adversarial disruption
        for i in 0..all_handles.len() {
            for j in (i + 1)..all_handles.len() {
                let (ha, hb) = (all_handles[i], all_handles[j]);
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
                    coop_events += 1;
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

        // Evolution: dead agents replaced by offspring of fittest same-type
        if tick > 0 && tick % EVOLUTION_INTERVAL == 0 {
            generation += 1;
            // Cooperator evolution
            let alive_coop: Vec<usize> = (0..COOPERATORS)
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&coop_handles[i])
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .collect();
            let dead_coop: Vec<usize> = (0..COOPERATORS)
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&coop_handles[i])
                        .map(|e| e.energy.is_collapsed())
                        .unwrap_or(true)
                })
                .collect();
            if !alive_coop.is_empty() {
                for &di in &dead_coop {
                    let parent = *alive_coop
                        .iter()
                        .max_by(|&&a, &&b| {
                            let ea = consciousness
                                .entities
                                .get(&coop_handles[a])
                                .map(|e| e.energy.available)
                                .unwrap_or(0.0);
                            let eb = consciousness
                                .entities
                                .get(&coop_handles[b])
                                .map(|e| e.energy.available)
                                .unwrap_or(0.0);
                            ea.partial_cmp(&eb).unwrap()
                        })
                        .unwrap();
                    let mut child_h = coop_harmonies[parent];
                    for k in 0..8 {
                        child_h[k] = (child_h[k] + (rng_f64(&mut rng) - 0.5) * MUTATION_RATE * 2.0)
                            .clamp(0.0, 1.0);
                    }
                    let h = coop_handles[di];
                    if let Some(e) = consciousness.entities.get_mut(&h) {
                        e.energy.available = consciousness.constants.initial_energy;
                        e.energy.collapsed = false;
                        e.harmony_activations = child_h;
                        e.prediction_error = 0.0;
                    }
                    if let Some(pb) = world.body(coop_handles[parent]) {
                        let pp = pb.position();
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
                    coop_harmonies[di] = child_h;
                    if coop_learn {
                        coop_weights[di] = coop_weights[parent].clone();
                    } // inherit learned weights
                }
            }
            // Adversary evolution (same pattern)
            let alive_adv: Vec<usize> = (0..ADVERSARIES)
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&adv_handles[i])
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .collect();
            let dead_adv: Vec<usize> = (0..ADVERSARIES)
                .filter(|&i| {
                    consciousness
                        .entities
                        .get(&adv_handles[i])
                        .map(|e| e.energy.is_collapsed())
                        .unwrap_or(true)
                })
                .collect();
            if !alive_adv.is_empty() {
                for &di in &dead_adv {
                    let parent = alive_adv[0];
                    let mut child_h = adv_harmonies[parent];
                    for k in 0..8 {
                        child_h[k] = (child_h[k] + (rng_f64(&mut rng) - 0.5) * MUTATION_RATE * 2.0)
                            .clamp(0.0, 1.0);
                    }
                    let h = adv_handles[di];
                    if let Some(e) = consciousness.entities.get_mut(&h) {
                        e.energy.available = consciousness.constants.initial_energy;
                        e.energy.collapsed = false;
                        e.harmony_activations = child_h;
                        e.prediction_error = 0.0;
                    }
                    if let Some(pb) = world.body(adv_handles[parent]) {
                        let pp = pb.position();
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
                    adv_harmonies[di] = child_h;
                    if adv_learn {
                        adv_weights[di] = adv_weights[parent].clone();
                    }
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    let coop_alive = coop_handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count() as f64;
    let adv_alive = adv_handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count() as f64;
    let coop_energy = coop_handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / COOPERATORS as f64;
    let adv_energy = adv_handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / ADVERSARIES.max(1) as f64;
    let coop_drift = coop_weights
        .iter()
        .map(|w| w.drift_from_default())
        .sum::<f64>()
        / COOPERATORS as f64;
    let adv_drift = adv_weights
        .iter()
        .map(|w| w.drift_from_default())
        .sum::<f64>()
        / ADVERSARIES.max(1) as f64;

    AdvLearnResult {
        condition: cond_name,
        coop_alive,
        adv_alive,
        coop_energy,
        adv_energy,
        coop_weight_drift: coop_drift,
        adv_weight_drift: adv_drift,
        cooperation_events: coop_events as f64,
        generations: generation as f64,
    }
}

fn main() {
    eprintln!("=== Adversarial Learning Experiment ===");
    eprintln!("Do cooperators need adaptation when adversaries evolve?");
    eprintln!(
        "{TOTAL} agents ({COOPERATORS} coop + {ADVERSARIES} adv), {TICKS} ticks, {SEEDS} seeds"
    );

    println!(
        "condition,seed,coop_alive,adv_alive,coop_energy,adv_energy,coop_drift,adv_drift,coop_events,generations"
    );

    let modes = [
        (Mode::FixedFixed, "FIXED_v_FIXED"),
        (Mode::LearnFixed, "LEARN_v_FIXED"),
        (Mode::LearnLearn, "LEARN_v_LEARN"),
    ];
    let mut all: Vec<(&str, Vec<AdvLearnResult>)> = Vec::new();

    for &(mode, name) in &modes {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(mode, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.1},{:.1},{:.4},{:.4},{:.0},{:.0}",
                r.condition,
                r.coop_alive,
                r.adv_alive,
                r.coop_energy,
                r.adv_energy,
                r.coop_weight_drift,
                r.adv_weight_drift,
                r.cooperation_events,
                r.generations
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: coop={:.1}/{COOPERATORS}, adv={:.1}/{ADVERSARIES}, drift={:.3}/{:.3}",
            results.iter().map(|r| r.coop_alive).sum::<f64>() / n,
            results.iter().map(|r| r.adv_alive).sum::<f64>() / n,
            results.iter().map(|r| r.coop_weight_drift).sum::<f64>() / n,
            results.iter().map(|r| r.adv_weight_drift).sum::<f64>() / n
        );
        all.push((name, results));
    }

    // Summary
    eprintln!("\n── Adversarial Learning ──");
    eprintln!("  Condition        Coop    Adv     CoopE   AdvE    CoopDrift  AdvDrift");
    for (name, results) in &all {
        let n = results.len() as f64;
        eprintln!(
            "  {:15} {:4.1}    {:4.1}    {:5.1}   {:5.1}   {:6.3}     {:6.3}",
            name,
            results.iter().map(|r| r.coop_alive).sum::<f64>() / n,
            results.iter().map(|r| r.adv_alive).sum::<f64>() / n,
            results.iter().map(|r| r.coop_energy).sum::<f64>() / n,
            results.iter().map(|r| r.adv_energy).sum::<f64>() / n,
            results.iter().map(|r| r.coop_weight_drift).sum::<f64>() / n,
            results.iter().map(|r| r.adv_weight_drift).sum::<f64>() / n
        );
    }

    // Key test: does learning help cooperators against learning adversaries?
    let ff_coop: Vec<f64> = all[0].1.iter().map(|r| r.coop_alive).collect();
    let ll_coop: Vec<f64> = all[2].1.iter().map(|r| r.coop_alive).collect();
    let lf_coop: Vec<f64> = all[1].1.iter().map(|r| r.coop_alive).collect();

    let tests = vec![
        ("FF vs LF (coop learns)", {
            let (_, _, p) = mann_whitney_u(&ff_coop, &lf_coop);
            p
        }),
        ("FF vs LL (both learn)", {
            let (_, _, p) = mann_whitney_u(&ff_coop, &ll_coop);
            p
        }),
        ("LF vs LL (adv learns too)", {
            let (_, _, p) = mann_whitney_u(&lf_coop, &ll_coop);
            p
        }),
    ];
    let effects = [
        cohens_d(&ff_coop, &lf_coop),
        cohens_d(&ff_coop, &ll_coop),
        cohens_d(&lf_coop, &ll_coop),
    ];
    let corrected = holm_bonferroni(&tests, 0.05);

    eprintln!("\n── Cooperator Survival (Holm-Bonferroni, k=3) ──");
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
            "  {label:25}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
            if sig { "← SIG" } else { "" }
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
