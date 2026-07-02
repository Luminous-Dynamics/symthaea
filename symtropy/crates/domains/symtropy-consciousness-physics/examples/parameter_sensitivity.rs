// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Parameter Sensitivity — are results fragile to constant tuning?
//!
//! Sweeps each ThermodynamicConstants::research() parameter independently
//! by ±50% to determine which parameters are critical vs irrelevant.
//! This directly addresses the "tuning leakage" concern: if results only
//! hold in a narrow parameter band, they're artifacts of tuning.
//!
//! Run: cargo run --example parameter_sensitivity --release

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

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

struct SensResult {
    alive: f64,
    clustering: f64,
    cooperation: f64,
}

fn run_with_constants(constants: ThermodynamicConstants, seed: u64) -> SensResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
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
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|&(_, &r)| r > 0.0)
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
            let pos = b.position();
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
        for &h in handles.iter() {
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

        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
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
        .filter_map(|&h| world.body(h).map(|b| b.position()))
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

    SensResult {
        alive,
        clustering,
        cooperation: coop as f64,
    }
}

fn sweep_parameter(
    name: &str,
    values: &[f64],
    mutator: impl Fn(&mut ThermodynamicConstants, f64),
) -> Vec<(f64, Vec<SensResult>)> {
    let mut all = Vec::new();
    for &val in values {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            let mut c = ThermodynamicConstants::research();
            mutator(&mut c, val);
            results.push(run_with_constants(c, seed));
        }
        all.push((val, results));
    }
    all
}

fn report_sweep(name: &str, results: &[(f64, Vec<SensResult>)]) {
    let baseline_idx = results.len() / 2; // middle value ≈ default
    let baseline_alive: Vec<f64> = results[baseline_idx].1.iter().map(|r| r.alive).collect();

    eprintln!("\n── {name} ──");
    eprintln!("  Value     Alive   Cluster  Coop     d(alive)");
    for (val, runs) in results {
        let n = runs.len() as f64;
        let alive = runs.iter().map(|r| r.alive).sum::<f64>() / n;
        let cluster = runs.iter().map(|r| r.clustering).sum::<f64>() / n;
        let coop = runs.iter().map(|r| r.cooperation).sum::<f64>() / n;
        let run_alive: Vec<f64> = runs.iter().map(|r| r.alive).collect();
        let d = cohens_d(&baseline_alive, &run_alive);
        let marker = if d.abs() > 0.8 { " ← SENSITIVE" } else { "" };
        eprintln!("  {val:7.4}  {alive:5.1}   {cluster:6.2}   {coop:8.0}  d={d:+.2}{marker}");
    }
}

fn main() {
    eprintln!("=== Parameter Sensitivity Analysis ===");
    eprintln!("Are results fragile to constant tuning?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds per value");

    println!("parameter,value,seed,alive,clustering,cooperation");

    // 1. Maintenance cost
    let maintenance_vals = [0.10, 0.15, 0.20, 0.25, 0.30];
    eprintln!("\nSweeping consciousness_maintenance_per_tick...");
    let r1 = sweep_parameter("maintenance", &maintenance_vals, |c, v| {
        c.consciousness_maintenance_per_tick = v
    });
    for (val, runs) in &r1 {
        for (s, r) in runs.iter().enumerate() {
            println!(
                "maintenance,{val},{},{:.1},{:.2},{:.0}",
                42 + s as u64 * 997,
                r.alive,
                r.clustering,
                r.cooperation
            );
        }
    }
    report_sweep("consciousness_maintenance_per_tick", &r1);

    // 2. Harmony range
    let range_vals = [20.0, 30.0, 40.0, 50.0, 60.0];
    eprintln!("\nSweeping harmony_range...");
    let r2 = sweep_parameter("harmony_range", &range_vals, |c, v| c.harmony_range = v);
    for (val, runs) in &r2 {
        for (s, r) in runs.iter().enumerate() {
            println!(
                "harmony_range,{val},{},{:.1},{:.2},{:.0}",
                42 + s as u64 * 997,
                r.alive,
                r.clustering,
                r.cooperation
            );
        }
    }
    report_sweep("harmony_range", &r2);

    // 3. Resonance regen rate
    let regen_vals = [0.03, 0.045, 0.06, 0.075, 0.09];
    eprintln!("\nSweeping harmony_resonance_regen_rate...");
    let r3 = sweep_parameter("resonance_regen", &regen_vals, |c, v| {
        c.harmony_resonance_regen_rate = v
    });
    for (val, runs) in &r3 {
        for (s, r) in runs.iter().enumerate() {
            println!(
                "resonance_regen,{val},{},{:.1},{:.2},{:.0}",
                42 + s as u64 * 997,
                r.alive,
                r.clustering,
                r.cooperation
            );
        }
    }
    report_sweep("harmony_resonance_regen_rate", &r3);

    // 4. Ambient regen
    let ambient_vals = [0.000, 0.0025, 0.005, 0.0075, 0.01];
    eprintln!("\nSweeping ambient_regen_rate...");
    let r4 = sweep_parameter("ambient_regen", &ambient_vals, |c, v| {
        c.ambient_regen_rate = v
    });
    for (val, runs) in &r4 {
        for (s, r) in runs.iter().enumerate() {
            println!(
                "ambient_regen,{val},{},{:.1},{:.2},{:.0}",
                42 + s as u64 * 997,
                r.alive,
                r.clustering,
                r.cooperation
            );
        }
    }
    report_sweep("ambient_regen_rate", &r4);

    // Summary: which parameters are critical?
    eprintln!("\n── SENSITIVITY SUMMARY ──");
    let params = [
        ("maintenance", &r1),
        ("harmony_range", &r2),
        ("resonance_regen", &r3),
        ("ambient_regen", &r4),
    ];
    for (name, results) in &params {
        let baseline_idx = results.len() / 2;
        let baseline_alive: Vec<f64> = results[baseline_idx].1.iter().map(|r| r.alive).collect();
        let max_d = results
            .iter()
            .map(|(_val, runs)| {
                let alive: Vec<f64> = runs.iter().map(|r| r.alive).collect();
                cohens_d(&baseline_alive, &alive).abs()
            })
            .fold(0.0f64, f64::max);
        let verdict = if max_d > 0.8 {
            "CRITICAL"
        } else if max_d > 0.5 {
            "moderate"
        } else {
            "robust"
        };
        eprintln!("  {name:20}: max |d| = {max_d:.2} → {verdict}");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
