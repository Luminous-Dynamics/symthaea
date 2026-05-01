// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bowling Alone — does declining social infrastructure produce Putnam's pattern?
//!
//! Robert Putnam (2000) documented 4 simultaneous trends in American society:
//! 1. Declining civic participation (cooperation events)
//! 2. Increasing social isolation (clustering loosens)
//! 3. Persistent economic output (survival doesn't immediately drop)
//! 4. Rising inequality (Gini increases)
//!
//! We model this by GRADUALLY reducing harmony_range (social infrastructure)
//! and resonance_regen (mutual benefit) over time — simulating 5 "decades"
//! of social capital erosion. The question: does the engine reproduce all
//! 4 Putnam trends simultaneously, or only some?
//!
//! Compares:
//! 1. STABLE: constant social infrastructure
//! 2. ERODING: range and regen decline 10% per decade (50% total)
//! 3. COLLAPSED: range and regen decline 20% per decade (67% total)
//!
//! Run: cargo run --example bowling_alone --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 24;
const TICKS: usize = 15_000; // 5 decades of 3000 ticks each
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;
const DECADE_TICKS: usize = 3000;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy)]
enum ErosionMode {
    Stable,
    Eroding,
    Collapsed,
}

struct DecadeSnapshot {
    decade: usize,
    alive: f64,
    cooperation_rate: f64, // coop events per tick this decade
    clustering: f64,
    gini: f64,
    current_range: f64,
    current_regen: f64,
}

struct BowlingResult {
    condition: &'static str,
    decades: Vec<DecadeSnapshot>,
    final_alive: f64,
    putnam_score: f64, // how many of the 4 trends are reproduced
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

fn run_experiment(erosion: ErosionMode, seed: u64) -> BowlingResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let base_range = consciousness.constants.harmony_range; // 40.0
    let base_regen = consciousness.constants.harmony_resonance_regen_rate; // 0.06

    let decay_per_decade: f64 = match erosion {
        ErosionMode::Stable => 0.0,
        ErosionMode::Eroding => 0.10,   // 10% per decade
        ErosionMode::Collapsed => 0.20, // 20% per decade
    };

    let wells = vec![
        SVector::from([40.0, 10.0]),
        SVector::from([-30.0, -20.0]),
        SVector::from([10.0, -40.0]),
    ];
    let mut well_remaining = vec![5000.0f64; 3]; // large wells for long run

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 100.0;
        let y = (rng_f64(&mut rng) - 0.5) * 100.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.05;
        }
        consciousness.register(h, consciousness.constants.initial_energy, base_range);
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = HARMONY_PROFILES[i % 4];
        }
        handles.push(h);
    }

    let cond_name = match erosion {
        ErosionMode::Stable => "STABLE",
        ErosionMode::Eroding => "ERODING",
        ErosionMode::Collapsed => "COLLAPSED",
    };

    let mut decades = Vec::new();
    let mut decade_coop = 0u64;
    let mut current_range = base_range;
    let mut current_regen = base_regen;

    for tick in 0..TICKS {
        let decade = tick / DECADE_TICKS;

        // Erode social infrastructure at decade boundaries
        if tick > 0 && tick % DECADE_TICKS == 0 {
            // Snapshot the ending decade
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

            decades.push(DecadeSnapshot {
                decade: decade - 1,
                alive,
                cooperation_rate: decade_coop as f64 / DECADE_TICKS as f64,
                clustering: if clustering.is_finite() {
                    clustering
                } else {
                    0.0
                },
                gini: gini(&energies),
                current_range,
                current_regen,
            });
            decade_coop = 0;

            // Apply erosion
            current_range = (base_range * (1.0 - decay_per_decade).powi(decade as i32)).max(5.0);
            current_regen = (base_regen * (1.0 - decay_per_decade).powi(decade as i32)).max(0.001);
            consciousness.constants.harmony_range = current_range;
            consciousness.constants.harmony_resonance_regen_rate = current_regen;
        }

        // Standard simulation loop
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
            .map(|(&p, &r)| (p, (r / 5000.0).min(1.0)))
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
                    d > 2.0 && d < current_range
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
                    (Some(a), Some(b)) => a.position().distance(b.position()) < current_range,
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
                    let rg = current_regen * (res - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.regenerate(rg);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.regenerate(rg);
                    }
                    decade_coop += 1;
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    // Final decade snapshot
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
        0.0
    };
    decades.push(DecadeSnapshot {
        decade: TICKS / DECADE_TICKS,
        alive,
        cooperation_rate: decade_coop as f64 / DECADE_TICKS as f64,
        clustering: if clustering.is_finite() {
            clustering
        } else {
            0.0
        },
        gini: gini(&energies),
        current_range,
        current_regen,
    });

    // Count Putnam trends (comparing decade 0 vs last decade)
    let mut putnam = 0.0;
    if decades.len() >= 2 {
        let first = &decades[0];
        let last = &decades[decades.len() - 1];
        if last.cooperation_rate < first.cooperation_rate * 0.8 {
            putnam += 1.0;
        } // declining cooperation
        if last.clustering > first.clustering * 1.2 || last.clustering == 0.0 {
            putnam += 1.0;
        } // increasing isolation
        if last.alive > first.alive * 0.7 {
            putnam += 1.0;
        } // persistent survival (economy)
        if last.gini > first.gini + 0.05 {
            putnam += 1.0;
        } // rising inequality
    }

    BowlingResult {
        condition: cond_name,
        decades,
        final_alive: alive,
        putnam_score: putnam,
    }
}

fn main() {
    eprintln!("=== Bowling Alone Experiment ===");
    eprintln!("Does declining social infrastructure produce Putnam's 4 trends?");
    eprintln!("{AGENTS} agents, {TICKS} ticks (5 decades), {SEEDS} seeds");
    eprintln!("Putnam trends: ↓cooperation, ↑isolation, →survival, ↑inequality");

    println!("condition,seed,decade,alive,coop_rate,clustering,gini,range,regen");

    let modes = [
        (ErosionMode::Stable, "STABLE"),
        (ErosionMode::Eroding, "ERODING"),
        (ErosionMode::Collapsed, "COLLAPSED"),
    ];
    let mut all: Vec<(&str, Vec<BowlingResult>)> = Vec::new();

    for &(mode, name) in &modes {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(mode, seed);
            for d in &r.decades {
                println!(
                    "{name},{seed},{},{:.1},{:.1},{:.2},{:.4},{:.1},{:.4}",
                    d.decade,
                    d.alive,
                    d.cooperation_rate,
                    d.clustering,
                    d.gini,
                    d.current_range,
                    d.current_regen
                );
            }
            results.push(r);
        }
        let n = results.len() as f64;
        let mean_putnam = results.iter().map(|r| r.putnam_score).sum::<f64>() / n;
        eprintln!("  → {name}: Putnam score = {mean_putnam:.2}/4.0");
        all.push((name, results));
    }

    // Decade-by-decade trajectory for ERODING
    eprintln!("\n── ERODING Trajectory (mean across {SEEDS} seeds) ──");
    eprintln!("  Decade  Range  Regen   Alive  CoopRate  Cluster  Gini");
    if let Some((_, results)) = all.iter().find(|(n, _)| *n == "ERODING") {
        let n_decades = results[0].decades.len();
        for di in 0..n_decades {
            let n = results.len() as f64;
            let alive = results.iter().map(|r| r.decades[di].alive).sum::<f64>() / n;
            let coop = results
                .iter()
                .map(|r| r.decades[di].cooperation_rate)
                .sum::<f64>()
                / n;
            let clust = results
                .iter()
                .filter_map(|r| {
                    let c = r.decades[di].clustering;
                    if c.is_finite() && c < 1e10 {
                        Some(c)
                    } else {
                        None
                    }
                })
                .sum::<f64>()
                / n;
            let gini = results.iter().map(|r| r.decades[di].gini).sum::<f64>() / n;
            let range = results[0].decades[di].current_range;
            let regen = results[0].decades[di].current_regen;
            eprintln!(
                "  {:5}   {:4.1}   {:.4}  {:5.1}  {:7.1}   {:6.2}   {:.3}",
                di, range, regen, alive, coop, clust, gini
            );
        }
    }

    // Putnam score summary
    eprintln!("\n── Putnam Score (trends reproduced out of 4) ──");
    for (name, results) in &all {
        let n = results.len() as f64;
        let score = results.iter().map(|r| r.putnam_score).sum::<f64>() / n;
        let marker = match score as usize {
            4 => " ← ALL 4 TRENDS",
            3 => " ← 3 of 4",
            _ => "",
        };
        eprintln!("  {:10}: {score:.2}/4.0{marker}", name);
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
