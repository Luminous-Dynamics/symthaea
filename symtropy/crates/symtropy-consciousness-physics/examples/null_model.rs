// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Null Model — do observed effects require specific mechanisms?
//!
//! Three null conditions establish that cooperation requires both FEP
//! gradient AND harmony resonance — not just agent movement.
//!
//! 1. BASELINE: Full system (FEP gradient + harmony resonance)
//! 2. RANDOM_WALK: Replace FEP gradient with uniform random direction
//! 3. SHUFFLED: Each tick, randomly reassign harmony profiles
//! 4. NO_REGEN: Disable harmony resonance regeneration entirely
//!
//! If cooperation vanishes under null conditions, the specific mechanisms
//! (FEP gradient, harmony resonance) are causally necessary.
//!
//! Run: cargo run --example null_model --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 8_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy, PartialEq)]
enum NullCondition {
    Baseline,
    RandomWalk,
    Shuffled,
    NoRegen,
}

struct NullResult {
    condition: &'static str,
    alive: f64,
    clustering: f64,
    cooperation: f64,
}

fn run_experiment(condition: NullCondition, seed: u64) -> NullResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![2500.0f64; 2];

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
            e.harmony_activations = HARMONY_PROFILES[i % 4];
        }
        handles.push(h);
    }

    let cond_name = match condition {
        NullCondition::Baseline => "BASELINE",
        NullCondition::RandomWalk => "RANDOM_WALK",
        NullCondition::Shuffled => "SHUFFLED",
        NullCondition::NoRegen => "NO_REGEN",
    };

    let mut coop = 0u64;

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

        // SHUFFLED: randomize harmony assignments each tick
        if condition == NullCondition::Shuffled {
            for &h in &handles {
                if let Some(e) = consciousness.entities.get_mut(&h) {
                    let idx = (rng_f64(&mut rng) * 4.0) as usize % 4;
                    e.harmony_activations = HARMONY_PROFILES[idx];
                }
            }
        }

        // Movement
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

        for &h in &handles {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position().0;

            let dir = match condition {
                NullCondition::RandomWalk => {
                    // Random direction (deterministic from position for reproducibility)
                    let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                    SVector::from([angle.cos(), angle.sin()])
                }
                _ => {
                    let near: Vec<_> = adata
                        .iter()
                        .filter(|(p, _)| {
                            let d = (p - pos).norm();
                            d > 2.0 && d < consciousness.constants.harmony_range
                        })
                        .cloned()
                        .collect();
                    fep_gradient::free_energy_gradient(
                        &pos,
                        e.energy.fraction_remaining(),
                        &e.harmony_activations,
                        &near,
                        &wdata,
                        None,
                        0.0,
                    )
                }
            };
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = dir * 20.0;
            }
        }

        // Maintenance + regen
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                if let Some(b) = world.body(h) {
                    let pos = b.position().0;
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

        // Cooperation (disabled for NO_REGEN)
        if condition != NullCondition::NoRegen {
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    let (ha, hb) = (handles[i], handles[j]);
                    let in_range = match (world.body(ha), world.body(hb)) {
                        (Some(a), Some(b)) => {
                            a.position().distance(b.position())
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
                        let rg = consciousness.constants.harmony_resonance_regen_rate
                            * (res - 0.5)
                            * 2.0;
                        if let Some(e) = consciousness.entities.get_mut(&ha) {
                            e.energy.regenerate(rg);
                        }
                        if let Some(e) = consciousness.entities.get_mut(&hb) {
                            e.energy.regenerate(rg);
                        }
                        coop += 1;
                    }
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
    let pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|h| world.body(*h).map(|b| b.position().0))
        .collect();
    let clustering = if pos.len() >= 2 {
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
    } else {
        f64::MAX
    };

    NullResult {
        condition: cond_name,
        alive,
        clustering,
        cooperation: coop as f64,
    }
}

fn main() {
    eprintln!("=== Null Model Experiment ===");
    eprintln!("Do observed effects require specific mechanisms?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds per condition");

    println!("condition,seed,alive,clustering,cooperation");

    let conditions = [
        (NullCondition::Baseline, "BASELINE"),
        (NullCondition::RandomWalk, "RANDOM_WALK"),
        (NullCondition::Shuffled, "SHUFFLED"),
        (NullCondition::NoRegen, "NO_REGEN"),
    ];
    let mut all_results: Vec<(&str, Vec<NullResult>)> = Vec::new();

    for &(cond, name) in &conditions {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(cond, seed);
            println!(
                "{},{seed},{:.1},{:.2},{:.0}",
                r.condition, r.alive, r.clustering, r.cooperation
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, cluster={:.2}, coop={:.0}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n
        );
        all_results.push((name, results));
    }

    // Summary table
    eprintln!("\n── Null Model Results ──");
    eprintln!("  Condition      Alive   Cluster  Cooperation");
    for (name, results) in &all_results {
        let n = results.len() as f64;
        eprintln!(
            "  {:13} {:5.1}   {:6.2}   {:10.0}",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n
        );
    }

    // Statistical tests: BASELINE vs each null (Holm-Bonferroni corrected)
    let baseline = &all_results[0].1;
    let b_alive: Vec<f64> = baseline.iter().map(|r| r.alive).collect();

    let mut p_values = Vec::new();
    let mut effects = Vec::new();
    for (name, results) in &all_results[1..] {
        let n_alive: Vec<f64> = results.iter().map(|r| r.alive).collect();
        let (_, _, p) = mann_whitney_u(&b_alive, &n_alive);
        let d = cohens_d(&b_alive, &n_alive);
        p_values.push((*name, p));
        effects.push(d);
    }

    let corrected = holm_bonferroni(&p_values, 0.05);
    eprintln!("\n── BASELINE vs Null Conditions (Holm-Bonferroni, k=3) ──");
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
            "  vs {:13}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
            label,
            if sig { "← SIGNIFICANT" } else { "" }
        );
    }

    let all_sig = corrected.iter().all(|r| r.2);
    let any_sig = corrected.iter().any(|r| r.2);
    eprintln!(
        "\n  Verdict: {}",
        if all_sig {
            "ALL NULL MODELS DIFFER — mechanisms are causally necessary"
        } else if any_sig {
            "SOME null models differ — partial causal specificity"
        } else {
            "NO significant differences — cooperation may be an artifact"
        }
    );

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
