// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! U-Curve Diagnostic — WHY is w=0.6 worse than w=0.0?
//!
//! Finding 43 showed a U-shaped survival curve: full cooperation (w=1.0)
//! and full selfishness (w=0.0) both outperform partial cooperation (w=0.6).
//!
//! Hypothesis: partial cooperators waste energy seeking partners they then
//! ignore, getting the cost of movement without the benefit of resonance.
//!
//! This diagnostic tracks per-tick energy flows:
//! - Energy spent on movement (displacement × cost)
//! - Energy gained from resonance cooperation
//! - Energy gained from wells
//! - Net energy balance per agent per tick
//! - Time spent "seeking" (moving toward agents) vs "gathering" (at wells)
//!
//! Run: cargo run --example ucurve_diagnostic --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 8_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;

const WILLINGNESS_LEVELS: [f64; 5] = [1.0, 0.6, 0.4, 0.2, 0.0];

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

struct DiagResult {
    willingness: f64,
    alive: f64,
    total_maintenance: f64,     // energy spent on consciousness maintenance
    total_well_regen: f64,      // energy gained from wells
    total_resonance_regen: f64, // energy gained from cooperation
    total_ambient_regen: f64,   // energy from ambient
    ticks_at_well: f64,         // proportion of agent-ticks spent near a well
    ticks_seeking: f64,         // proportion moving toward agents (social gradient active)
    mean_displacement: f64,     // average distance traveled per tick
    energy_efficiency: f64,     // regen / maintenance ratio
}

fn run_diagnostic(willingness: f64, seed: u64) -> DiagResult {
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

    let mut total_maintenance = 0.0f64;
    let mut total_well_regen = 0.0f64;
    let mut total_resonance_regen = 0.0f64;
    let mut total_ambient_regen = 0.0f64;
    let mut ticks_at_well = 0u64;
    let mut ticks_seeking = 0u64;
    let mut total_displacement = 0.0f64;
    let mut agent_tick_count = 0u64;

    let mut prev_positions: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position().0))
        .collect();

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

            let include_social = rng_f64(&mut rng) < willingness;
            if include_social {
                ticks_seeking += 1;
            }
            agent_tick_count += 1;

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
                vec![]
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

        // Track displacement
        let current_pos: Vec<SVector<f64, 2>> = handles
            .iter()
            .filter_map(|&h| world.body(h).map(|b| b.position().0))
            .collect();
        if current_pos.len() == prev_positions.len() {
            for i in 0..current_pos.len() {
                total_displacement += (current_pos[i] - prev_positions[i]).norm();
            }
        }
        prev_positions = current_pos;

        // Maintenance
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            let maint = mr * (1.0 + consciousness.phi(h) * 0.5);
            consciousness.consume_energy(h, maint);
            total_maintenance += maint;

            if let Some(e) = consciousness.entities.get_mut(&h) {
                let amb = ar * rm;
                e.energy.regenerate(amb);
                total_ambient_regen += amb;

                if let Some(b) = world.body(h) {
                    let pos = b.position().0;
                    for (wi, &w) in wells.iter().enumerate() {
                        if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                            let d = wr.min(well_remaining[wi]);
                            e.energy.regenerate(d);
                            well_remaining[wi] -= d;
                            total_well_regen += d;
                            ticks_at_well += 1;
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
                    total_resonance_regen += rg * 2.0; // both agents get it
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
    let total_regen = total_well_regen + total_resonance_regen + total_ambient_regen;

    DiagResult {
        willingness,
        alive,
        total_maintenance,
        total_well_regen,
        total_resonance_regen,
        total_ambient_regen,
        ticks_at_well: if agent_tick_count > 0 {
            ticks_at_well as f64 / agent_tick_count as f64
        } else {
            0.0
        },
        ticks_seeking: if agent_tick_count > 0 {
            ticks_seeking as f64 / agent_tick_count as f64
        } else {
            0.0
        },
        mean_displacement: total_displacement / (AGENTS as f64 * TICKS as f64),
        energy_efficiency: if total_maintenance > 0.0 {
            total_regen / total_maintenance
        } else {
            0.0
        },
    }
}

fn main() {
    eprintln!("=== U-Curve Diagnostic ===");
    eprintln!("WHY is w=0.6 worse than w=0.0?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    println!("willingness,seed,alive,maintenance,well_regen,resonance_regen,ambient_regen,at_well_frac,seeking_frac,displacement,efficiency");

    let mut all: Vec<(f64, Vec<DiagResult>)> = Vec::new();

    for &w in &WILLINGNESS_LEVELS {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  w={w:.1} seed={seed}...");
            let r = run_diagnostic(w, seed);
            println!(
                "{w:.1},{seed},{:.1},{:.0},{:.0},{:.0},{:.0},{:.3},{:.3},{:.3},{:.3}",
                r.alive,
                r.total_maintenance,
                r.total_well_regen,
                r.total_resonance_regen,
                r.total_ambient_regen,
                r.ticks_at_well,
                r.ticks_seeking,
                r.mean_displacement,
                r.energy_efficiency
            );
            results.push(r);
        }
        all.push((w, results));
    }

    eprintln!("\n── Energy Flow Diagnostic ──");
    eprintln!(
        "  Will  Alive  Maint     Wells     Reson     Ambient   AtWell  Seeking  Displ  Efficiency"
    );
    for (w, results) in &all {
        let n = results.len() as f64;
        eprintln!("  {w:.1}  {:5.1}  {:7.0}   {:7.0}   {:7.0}   {:7.0}   {:5.1}%  {:5.1}%   {:.2}   {:.3}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.total_maintenance).sum::<f64>() / n,
            results.iter().map(|r| r.total_well_regen).sum::<f64>() / n,
            results.iter().map(|r| r.total_resonance_regen).sum::<f64>() / n,
            results.iter().map(|r| r.total_ambient_regen).sum::<f64>() / n,
            results.iter().map(|r| r.ticks_at_well).sum::<f64>() / n * 100.0,
            results.iter().map(|r| r.ticks_seeking).sum::<f64>() / n * 100.0,
            results.iter().map(|r| r.mean_displacement).sum::<f64>() / n,
            results.iter().map(|r| r.energy_efficiency).sum::<f64>() / n);
    }

    // Mechanism analysis
    eprintln!("\n── U-Curve Mechanism ──");
    let w10 = &all[0]; // w=1.0
    let w06 = &all[1]; // w=0.6
    let w00 = &all[4]; // w=0.0
    let n = SEEDS as f64;

    let well_10 = w10.1.iter().map(|r| r.total_well_regen).sum::<f64>() / n;
    let well_06 = w06.1.iter().map(|r| r.total_well_regen).sum::<f64>() / n;
    let well_00 = w00.1.iter().map(|r| r.total_well_regen).sum::<f64>() / n;
    let res_10 = w10.1.iter().map(|r| r.total_resonance_regen).sum::<f64>() / n;
    let res_06 = w06.1.iter().map(|r| r.total_resonance_regen).sum::<f64>() / n;
    let res_00 = w00.1.iter().map(|r| r.total_resonance_regen).sum::<f64>() / n;
    let eff_10 = w10.1.iter().map(|r| r.energy_efficiency).sum::<f64>() / n;
    let eff_06 = w06.1.iter().map(|r| r.energy_efficiency).sum::<f64>() / n;
    let eff_00 = w00.1.iter().map(|r| r.energy_efficiency).sum::<f64>() / n;

    eprintln!("  w=1.0: wells={well_10:.0}J, resonance={res_10:.0}J, efficiency={eff_10:.3}");
    eprintln!("  w=0.6: wells={well_06:.0}J, resonance={res_06:.0}J, efficiency={eff_06:.3}");
    eprintln!("  w=0.0: wells={well_00:.0}J, resonance={res_00:.0}J, efficiency={eff_00:.3}");

    if well_00 > well_06 && res_10 > res_06 {
        eprintln!("\n  CONFIRMED: w=0.6 gets LESS well energy than w=0.0 (distracted by");
        eprintln!("  social seeking) AND less resonance than w=1.0 (inconsistent commitment).");
        eprintln!(
            "  Partial cooperation pays the cost of both strategies but reaps neither benefit."
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
