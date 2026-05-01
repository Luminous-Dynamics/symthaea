// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Curvature Cooperation — does conformal geometry improve cooperation?
//!
//! Tests whether integration-parameterized curvature (g_ij = e^{2σ}δ_ij)
//! creates "harmony wells" that enhance agent clustering and survival.
//! If curvature matters for cooperation (not just trajectory deflection),
//! it justifies keeping the feature in the engine.
//!
//! 3 conditions: NO_CURVATURE, LOW (κ=0.01), HIGH (κ=0.05)
//!
//! Run: cargo run --example curvature_cooperation --release --features consciousness-curvature

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

#[cfg(feature = "consciousness-curvature")]
use symtropy_consciousness_physics::curvature::ConformalMetric;

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

struct CurvResult {
    condition: &'static str,
    alive: f64,
    energy: f64,
    clustering: f64,
    cooperation: f64,
}

fn run_experiment(curvature_scale: f64, seed: u64) -> CurvResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![2500.0f64; 2];

    let mut rng = seed;
    let mut handles = Vec::new();

    // Place harmony sources for curvature field
    #[cfg(feature = "consciousness-curvature")]
    let metric = ConformalMetric::<2>::with_scale(curvature_scale);

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

    let cond_name = if curvature_scale < 1e-6 {
        "FLAT"
    } else if curvature_scale < 0.02 {
        "LOW_CURV"
    } else {
        "HIGH_CURV"
    };

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
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();

            let mut vel = fep_gradient::free_energy_gradient(
                &pos,
                e.energy.fraction_remaining(),
                &e.harmony_activations,
                &near,
                &wdata,
                None,
                0.0,
            ) * 20.0;

            // Apply geodesic correction from conformal curvature
            #[cfg(feature = "consciousness-curvature")]
            if curvature_scale > 1e-6 {
                // Compute sigma gradient from harmony field
                let harmony_energy: f64 = near
                    .iter()
                    .map(|(p, h)| {
                        let d = (p - pos).norm().max(1.0);
                        let strength: f64 = h.iter().sum::<f64>() / 8.0;
                        strength / d
                    })
                    .sum();
                let sigma_grad = {
                    let mut g = SVector::<f64, 2>::zeros();
                    for (p, h) in &near {
                        let delta = p - pos;
                        let d = delta.norm().max(1.0);
                        let strength: f64 = h.iter().sum::<f64>() / 8.0;
                        g -= delta / (d * d * d) * strength * curvature_scale;
                    }
                    g
                };
                let correction = metric.geodesic_correction(&vel, &sigma_grad);
                vel += correction * DT;
            }

            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = vel;
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

    CurvResult {
        condition: cond_name,
        alive,
        energy,
        clustering,
        cooperation: coop as f64,
    }
}

fn main() {
    eprintln!("=== Curvature Cooperation Experiment ===");
    eprintln!("Does conformal geometry improve cooperation?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    #[cfg(not(feature = "consciousness-curvature"))]
    {
        eprintln!("WARNING: consciousness-curvature feature not enabled!");
        eprintln!("Run with: cargo run --example curvature_cooperation --release --features consciousness-curvature");
        eprintln!("Running FLAT-only comparison...");
    }

    println!("condition,seed,alive,energy,clustering,cooperation");

    let scales = [(0.0, "FLAT"), (0.01, "LOW_CURV"), (0.05, "HIGH_CURV")];
    let mut all_results: Vec<(&str, Vec<CurvResult>)> = Vec::new();

    for &(scale, name) in &scales {
        #[cfg(not(feature = "consciousness-curvature"))]
        if scale > 0.0 {
            eprintln!("  Skipping {name} (no curvature feature)");
            continue;
        }

        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(scale, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.2},{:.0}",
                r.condition, r.alive, r.energy, r.clustering, r.cooperation
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, cluster={:.2}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n
        );
        all_results.push((name, results));
    }

    if all_results.len() >= 2 {
        let flat = &all_results[0].1;
        let mut p_values = Vec::new();
        let mut effects = Vec::new();
        let f_alive: Vec<f64> = flat.iter().map(|r| r.alive).collect();

        for (name, results) in &all_results[1..] {
            let n_alive: Vec<f64> = results.iter().map(|r| r.alive).collect();
            let (_, _, p) = mann_whitney_u(&f_alive, &n_alive);
            let d = cohens_d(&f_alive, &n_alive);
            p_values.push((*name, p));
            effects.push(d);
        }

        let corrected = holm_bonferroni(&p_values, 0.05);
        eprintln!("\n── FLAT vs Curvature (Holm-Bonferroni) ──");
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
                "  vs {:10}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
                label,
                if sig { "← SIG" } else { "" }
            );
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
