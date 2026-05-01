// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Volitional Cooperation — what happens when agents CAN refuse?
//!
//! Finding 42 showed the engine can't reproduce Putnam's "Bowling Alone"
//! because agents are thermodynamic automatons who MUST cooperate.
//! This experiment adds a single parameter: `cooperation_willingness`,
//! the probability an agent includes the social component in its FEP gradient.
//!
//! At willingness=1.0, agents always seek resonant partners (baseline).
//! At willingness=0.0, agents ignore other agents entirely (pure well-seeking).
//! Between: agents probabilistically include or exclude the social gradient.
//!
//! Tests whether adding minimal volition:
//! 1. Allows Putnam-style cooperation decline
//! 2. Creates a willingness threshold below which cooperation collapses
//! 3. Changes the topology of social structure
//!
//! Run: cargo run --example volitional_cooperation --release

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

const WILLINGNESS_LEVELS: [f64; 7] = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0];

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

struct VolitionResult {
    willingness: f64,
    alive: f64,
    energy: f64,
    clustering: f64,
    cooperation: f64,
    gini: f64,
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

fn run_experiment(willingness: f64, seed: u64) -> VolitionResult {
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

    let mut coop = 0u64;

    for _tick in 0..TICKS {
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

            // VOLITION: each tick, roll whether to include social gradient
            let include_social = rng_f64(&mut rng) < willingness;

            let near: Vec<_> = if include_social {
                adata
                    .iter()
                    .filter(|(p, _)| {
                        let d = (p - pos).norm();
                        d > 2.0 && d < consciousness.constants.harmony_range
                    })
                    .cloned()
                    .collect()
            } else {
                vec![] // ignore other agents entirely
            };

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

        // Cooperation still happens when agents are near (resonance is passive)
        // But willingness affects whether agents SEEK each other
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
                    coop += 1;
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

    VolitionResult {
        willingness,
        alive,
        energy,
        clustering: if clustering.is_finite() {
            clustering
        } else {
            0.0
        },
        cooperation: coop as f64,
        gini: gini(&energies),
    }
}

fn main() {
    eprintln!("=== Volitional Cooperation Experiment ===");
    eprintln!("What happens when agents CAN refuse to cooperate?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("Willingness = probability of including social gradient");

    println!("willingness,seed,alive,energy,clustering,cooperation,gini");

    let mut all: Vec<(f64, Vec<VolitionResult>)> = Vec::new();

    for &w in &WILLINGNESS_LEVELS {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  w={w:.1} seed={seed}...");
            let r = run_experiment(w, seed);
            println!(
                "{w:.1},{seed},{:.1},{:.1},{:.2},{:.0},{:.4}",
                r.alive, r.energy, r.clustering, r.cooperation, r.gini
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → w={w:.1}: alive={:.1}, coop={:.0}, gini={:.3}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n,
            results.iter().map(|r| r.gini).sum::<f64>() / n
        );
        all.push((w, results));
    }

    eprintln!("\n── Willingness Gradient ──");
    eprintln!("  Will   Alive  Coop       Cluster  Gini    Putnam?");
    let mut critical_w = None;
    for (w, results) in &all {
        let n = results.len() as f64;
        let alive = results.iter().map(|r| r.alive).sum::<f64>() / n;
        let coop = results.iter().map(|r| r.cooperation).sum::<f64>() / n;
        let clust = results.iter().map(|r| r.clustering).sum::<f64>() / n;
        let gi = results.iter().map(|r| r.gini).sum::<f64>() / n;
        let survival_rate = alive / AGENTS as f64;
        let marker = if survival_rate < 0.5 && critical_w.is_none() {
            critical_w = Some(*w);
            " ← CRITICAL"
        } else {
            ""
        };
        eprintln!(
            "  {w:.1}    {:5.1}  {:9.0}   {:6.2}   {:.3}{marker}",
            alive, coop, clust, gi
        );
    }

    // Putnam analysis: compare w=1.0 (full cooperation) vs w=0.2 (low willingness)
    let full = &all[0].1; // w=1.0
    let low_idx = WILLINGNESS_LEVELS
        .iter()
        .position(|&w| w == 0.2)
        .unwrap_or(4);
    let low = &all[low_idx].1; // w=0.2

    let f_alive: Vec<f64> = full.iter().map(|r| r.alive).collect();
    let l_alive: Vec<f64> = low.iter().map(|r| r.alive).collect();
    let f_coop: Vec<f64> = full.iter().map(|r| r.cooperation).collect();
    let l_coop: Vec<f64> = low.iter().map(|r| r.cooperation).collect();
    let f_gini: Vec<f64> = full.iter().map(|r| r.gini).collect();
    let l_gini: Vec<f64> = low.iter().map(|r| r.gini).collect();

    let tests = vec![
        ("survival", {
            let (_, _, p) = mann_whitney_u(&f_alive, &l_alive);
            p
        }),
        ("cooperation", {
            let (_, _, p) = mann_whitney_u(&f_coop, &l_coop);
            p
        }),
        ("inequality", {
            let (_, _, p) = mann_whitney_u(&f_gini, &l_gini);
            p
        }),
    ];
    let effects = [
        cohens_d(&f_alive, &l_alive),
        cohens_d(&f_coop, &l_coop),
        cohens_d(&f_gini, &l_gini),
    ];
    let corrected = holm_bonferroni(&tests, 0.05);

    eprintln!("\n── w=1.0 vs w=0.2 (Holm-Bonferroni, k=3) ──");
    for (i, &(label, adj_p, sig)) in corrected.iter().enumerate() {
        let d = effects[i];
        let size = if d.abs() > 0.8 {
            "LARGE"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        };
        eprintln!(
            "  {label:12}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
            if sig { "← SIG" } else { "" }
        );
    }

    // Putnam check: does low willingness produce all 4 trends?
    let base_alive = f_alive.iter().sum::<f64>() / f_alive.len() as f64;
    let base_coop = f_coop.iter().sum::<f64>() / f_coop.len() as f64;
    let low_alive_mean = l_alive.iter().sum::<f64>() / l_alive.len() as f64;
    let low_coop_mean = l_coop.iter().sum::<f64>() / l_coop.len() as f64;
    let base_gini_mean = f_gini.iter().sum::<f64>() / f_gini.len() as f64;
    let low_gini_mean = l_gini.iter().sum::<f64>() / l_gini.len() as f64;

    let mut putnam = 0;
    if low_coop_mean < base_coop * 0.7 {
        putnam += 1;
        eprintln!("\n  ✓ Declining cooperation");
    }
    if low_alive_mean > base_alive * 0.5 {
        putnam += 1;
        eprintln!("  ✓ Persistent survival");
    }
    if low_gini_mean > base_gini_mean + 0.03 {
        putnam += 1;
        eprintln!("  ✓ Rising inequality");
    }
    // Isolation measured by clustering loosening
    let f_clust: Vec<f64> = full.iter().map(|r| r.clustering).collect();
    let l_clust: Vec<f64> = low.iter().map(|r| r.clustering).collect();
    let f_clust_mean = f_clust.iter().sum::<f64>() / f_clust.len() as f64;
    let l_clust_mean = l_clust.iter().sum::<f64>() / l_clust.len() as f64;
    if l_clust_mean > f_clust_mean * 1.2 {
        putnam += 1;
        eprintln!("  ✓ Increasing isolation");
    }

    eprintln!("\n  Putnam score: {putnam}/4");
    if putnam >= 3 {
        eprintln!("  VOLITION ENABLES BOWLING ALONE — adding choice reproduces Putnam's pattern");
    } else {
        eprintln!("  Volition alone insufficient — {putnam}/4 trends reproduced");
    }

    if let Some(cw) = critical_w {
        eprintln!("\n  Critical willingness: {cw:.1} — below this, >50% die");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
