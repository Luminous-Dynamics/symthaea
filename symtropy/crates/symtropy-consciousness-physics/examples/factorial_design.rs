// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 2³ Factorial Design — interaction effects between critical parameters.
//!
//! Based on the parameter sensitivity analysis (Finding 33), maintenance
//! and harmony_range are CRITICAL. This experiment tests all 8 combinations
//! of HIGH/LOW for the 3 most important parameters, revealing main effects
//! and 2-way interactions.
//!
//! Run: cargo run --example factorial_design --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::cohens_d;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 6_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;

// Factor levels (from sensitivity analysis)
const MAINT_LOW: f64 = 0.15;
const MAINT_HIGH: f64 = 0.30;
const RANGE_LOW: f64 = 25.0;
const RANGE_HIGH: f64 = 55.0;
const REGEN_LOW: f64 = 0.03;
const REGEN_HIGH: f64 = 0.09;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

struct FactResult {
    maint: &'static str,
    range: &'static str,
    regen: &'static str,
    alive: f64,
    clustering: f64,
    cooperation: f64,
}

fn run_condition(maint: f64, range: f64, regen: f64, seed: u64) -> (f64, f64, f64) {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = maint;
    constants.harmony_range = range;
    constants.harmony_resonance_regen_rate = regen;
    consciousness.constants = constants;

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
        consciousness.register(h, consciousness.constants.initial_energy, range);
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
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < range
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
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, maint * (1.0 + consciousness.phi(h) * 0.5));
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
                    (Some(a), Some(b)) => a.position().distance(b.position()) < range,
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
                    let rg = regen * (res - 0.5) * 2.0;
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

    (alive, clustering, coop as f64)
}

fn main() {
    eprintln!("=== 2³ Factorial Design ===");
    eprintln!("Interaction effects between critical parameters");
    eprintln!("Factors: maintenance ({MAINT_LOW}/{MAINT_HIGH}), range ({RANGE_LOW}/{RANGE_HIGH}), regen ({REGEN_LOW}/{REGEN_HIGH})");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds per condition");

    println!("maint,range,regen,seed,alive,clustering,cooperation");

    // 2³ = 8 conditions
    let conditions: [(f64, f64, f64, &str, &str, &str); 8] = [
        (MAINT_LOW, RANGE_LOW, REGEN_LOW, "L", "L", "L"),
        (MAINT_LOW, RANGE_LOW, REGEN_HIGH, "L", "L", "H"),
        (MAINT_LOW, RANGE_HIGH, REGEN_LOW, "L", "H", "L"),
        (MAINT_LOW, RANGE_HIGH, REGEN_HIGH, "L", "H", "H"),
        (MAINT_HIGH, RANGE_LOW, REGEN_LOW, "H", "L", "L"),
        (MAINT_HIGH, RANGE_LOW, REGEN_HIGH, "H", "L", "H"),
        (MAINT_HIGH, RANGE_HIGH, REGEN_LOW, "H", "H", "L"),
        (MAINT_HIGH, RANGE_HIGH, REGEN_HIGH, "H", "H", "H"),
    ];

    let mut all_alive: Vec<Vec<f64>> = Vec::new();

    for &(maint, range, regen, ml, rl, gl) in &conditions {
        let mut alive_vec = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  M={ml} R={rl} G={gl} seed={seed}...");
            let (alive, clust, coop) = run_condition(maint, range, regen, seed);
            println!("{ml},{rl},{gl},{seed},{alive:.1},{clust:.2},{coop:.0}");
            alive_vec.push(alive);
        }
        let n = alive_vec.len() as f64;
        let mean = alive_vec.iter().sum::<f64>() / n;
        eprintln!("  → M={ml} R={rl} G={gl}: alive={mean:.1}/{AGENTS}");
        all_alive.push(alive_vec);
    }

    // Compute main effects and interactions
    // Encoding: index bits = (maint_high, range_high, regen_high)
    let means: Vec<f64> = all_alive
        .iter()
        .map(|v| v.iter().sum::<f64>() / v.len() as f64)
        .collect();

    // Main effect of A (maintenance): mean(A=H) - mean(A=L)
    let a_high = (means[4] + means[5] + means[6] + means[7]) / 4.0;
    let a_low = (means[0] + means[1] + means[2] + means[3]) / 4.0;
    let main_a = a_high - a_low;

    // Main effect of B (range): mean(B=H) - mean(B=L)
    let b_high = (means[2] + means[3] + means[6] + means[7]) / 4.0;
    let b_low = (means[0] + means[1] + means[4] + means[5]) / 4.0;
    let main_b = b_high - b_low;

    // Main effect of C (regen): mean(C=H) - mean(C=L)
    let c_high = (means[1] + means[3] + means[5] + means[7]) / 4.0;
    let c_low = (means[0] + means[2] + means[4] + means[6]) / 4.0;
    let main_c = c_high - c_low;

    // AB interaction
    let ab_hh = (means[6] + means[7]) / 2.0;
    let ab_hl = (means[4] + means[5]) / 2.0;
    let ab_lh = (means[2] + means[3]) / 2.0;
    let ab_ll = (means[0] + means[1]) / 2.0;
    let interact_ab = (ab_hh - ab_hl) - (ab_lh - ab_ll);

    // AC interaction
    let ac_hh = (means[5] + means[7]) / 2.0;
    let ac_hl = (means[4] + means[6]) / 2.0;
    let ac_lh = (means[1] + means[3]) / 2.0;
    let ac_ll = (means[0] + means[2]) / 2.0;
    let interact_ac = (ac_hh - ac_hl) - (ac_lh - ac_ll);

    // BC interaction
    let bc_hh = (means[3] + means[7]) / 2.0;
    let bc_hl = (means[2] + means[6]) / 2.0;
    let bc_lh = (means[1] + means[5]) / 2.0;
    let bc_ll = (means[0] + means[4]) / 2.0;
    let interact_bc = (bc_hh - bc_hl) - (bc_lh - bc_ll);

    eprintln!("\n── 2³ Factorial Results ──");
    eprintln!("  Cell means (alive/{AGENTS}):");
    for (i, &(_, _, _, ml, rl, gl)) in conditions.iter().enumerate() {
        eprintln!("    M={ml} R={rl} G={gl}: {:.1}", means[i]);
    }

    eprintln!("\n── Main Effects (on survival) ──");
    eprintln!(
        "  A (maintenance HIGH-LOW):  {:+.2} agents {}",
        main_a,
        if main_a.abs() > 2.0 { "← STRONG" } else { "" }
    );
    eprintln!(
        "  B (range HIGH-LOW):        {:+.2} agents {}",
        main_b,
        if main_b.abs() > 2.0 { "← STRONG" } else { "" }
    );
    eprintln!(
        "  C (regen HIGH-LOW):        {:+.2} agents {}",
        main_c,
        if main_c.abs() > 2.0 { "← STRONG" } else { "" }
    );

    eprintln!("\n── 2-Way Interactions ──");
    eprintln!(
        "  A×B (maint × range):  {:+.2} {}",
        interact_ab,
        if interact_ab.abs() > 1.5 {
            "← INTERACTION"
        } else {
            ""
        }
    );
    eprintln!(
        "  A×C (maint × regen):  {:+.2} {}",
        interact_ac,
        if interact_ac.abs() > 1.5 {
            "← INTERACTION"
        } else {
            ""
        }
    );
    eprintln!(
        "  B×C (range × regen):  {:+.2} {}",
        interact_bc,
        if interact_bc.abs() > 1.5 {
            "← INTERACTION"
        } else {
            ""
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
