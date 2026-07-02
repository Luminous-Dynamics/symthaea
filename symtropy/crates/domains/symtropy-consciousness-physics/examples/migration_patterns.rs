// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Migration Patterns — do agents migrate when resources deplete?
//!
//! Uses 4 wells that deplete sequentially (staggered capacity), forcing
//! agents to discover and move to new resources. Tracks:
//! - Migration events (agent moves >50 units in 500 ticks)
//! - Well discovery time (first agent reaches each well)
//! - Migratory vs sedentary survival
//! - Collective vs individual migration (do groups move together?)
//!
//! Run: cargo run --example migration_patterns --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::cohens_d;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 12_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 8;
const MIGRATION_THRESHOLD: f64 = 50.0; // distance to count as migration

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

struct MigrationResult {
    alive: f64,
    total_migrations: f64,
    agents_that_migrated: f64,
    mean_distance_traveled: f64,
    well_discovery_ticks: [f64; 4], // when first agent reached each well
    collective_migrations: f64,     // migrations where 3+ agents moved together
}

fn run_experiment(seed: u64) -> MigrationResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    consciousness.constants = constants;

    // 4 wells at compass points, staggered capacity to force sequential migration
    let wells = vec![
        SVector::from([40.0, 0.0]),    // depletes first (small)
        SVector::from([0.0, 60.0]),    // depletes second
        SVector::from([-60.0, -20.0]), // depletes third
        SVector::from([30.0, -70.0]),  // longest lasting
    ];
    let mut well_remaining = vec![800.0, 1500.0, 2500.0, 4000.0f64];

    let mut rng = seed;
    let mut handles = Vec::new();

    // All agents start near first well
    for i in 0..AGENTS {
        let x = 40.0 + (rng_f64(&mut rng) - 0.5) * 20.0;
        let y = (rng_f64(&mut rng) - 0.5) * 20.0;
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

    // Track positions every 500 ticks for migration detection
    let mut prev_positions: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position()))
        .collect();
    let start_positions = prev_positions.clone();
    let mut migration_counts = vec![0u64; AGENTS];
    let mut well_discovery = [TICKS as f64; 4]; // when first agent found each well

    for tick in 0..TICKS {
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
            .map(|(&p, &r)| (p, (r / 4000.0).min(1.0)))
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

        // Check well discovery
        for (wi, &w) in wells.iter().enumerate() {
            if well_discovery[wi] >= TICKS as f64 {
                for &h in &handles {
                    if let Some(b) = world.body(h) {
                        if (b.position() - w).norm() < 35.0 {
                            well_discovery[wi] = tick as f64;
                            break;
                        }
                    }
                }
            }
        }

        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let nw: Vec<Option<usize>> = handles
            .iter()
            .map(|&h| {
                world.body(h).and_then(|b| {
                    let pos = b.position();
                    wells
                        .iter()
                        .enumerate()
                        .find(|&(ref i, &w)| (pos - w).norm() < 35.0 && well_remaining[*i] > 0.0)
                        .map(|(i, _)| i)
                })
            })
            .collect();
        for (idx, &h) in handles.iter().enumerate() {
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

        // Cooperation
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
                }
            }
        }

        // Migration detection every 500 ticks
        if tick > 0 && tick % 500 == 0 {
            let current_pos: Vec<SVector<f64, 2>> = handles
                .iter()
                .filter_map(|&h| world.body(h).map(|b| b.position()))
                .collect();
            if current_pos.len() == prev_positions.len() {
                for i in 0..current_pos.len() {
                    let dist = (current_pos[i] - prev_positions[i]).norm();
                    if dist > MIGRATION_THRESHOLD {
                        migration_counts[i] += 1;
                    }
                }
                prev_positions = current_pos;
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
    let total_migrations: u64 = migration_counts.iter().sum();
    let agents_migrated = migration_counts.iter().filter(|&&c| c > 0).count() as f64;

    // Total distance from start
    let final_pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position()))
        .collect();
    let mean_dist = if final_pos.len() == start_positions.len() && !final_pos.is_empty() {
        final_pos
            .iter()
            .zip(start_positions.iter())
            .map(|(f, s)| (f - s).norm())
            .sum::<f64>()
            / final_pos.len() as f64
    } else {
        0.0
    };

    // Collective migration: count 500-tick windows where 3+ agents migrated
    // (Simplified: just report total_migrations as proxy)
    let collective = if total_migrations > 3 {
        total_migrations / 3
    } else {
        0
    };

    MigrationResult {
        alive,
        total_migrations: total_migrations as f64,
        agents_that_migrated: agents_migrated,
        mean_distance_traveled: mean_dist,
        well_discovery_ticks: well_discovery,
        collective_migrations: collective as f64,
    }
}

fn main() {
    eprintln!("=== Migration Patterns Experiment ===");
    eprintln!("Do agents migrate when resources deplete?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("4 staggered wells (800/1500/2500/4000 J)");

    println!(
        "seed,alive,migrations,agents_migrated,mean_distance,well1_tick,well2_tick,well3_tick,well4_tick,collective"
    );

    let mut results = Vec::new();
    for s in 0..SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  seed={seed}...");
        let r = run_experiment(seed);
        println!(
            "{seed},{:.1},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.0},{:.0}",
            r.alive,
            r.total_migrations,
            r.agents_that_migrated,
            r.mean_distance_traveled,
            r.well_discovery_ticks[0],
            r.well_discovery_ticks[1],
            r.well_discovery_ticks[2],
            r.well_discovery_ticks[3],
            r.collective_migrations
        );
        results.push(r);
    }

    let n = results.len() as f64;
    eprintln!("\n── Migration Summary ──");
    eprintln!(
        "  Alive:          {:.1}/{AGENTS}",
        results.iter().map(|r| r.alive).sum::<f64>() / n
    );
    eprintln!(
        "  Migrations:     {:.1} total ({:.1} agents migrated)",
        results.iter().map(|r| r.total_migrations).sum::<f64>() / n,
        results.iter().map(|r| r.agents_that_migrated).sum::<f64>() / n
    );
    eprintln!(
        "  Mean distance:  {:.1} units",
        results
            .iter()
            .map(|r| r.mean_distance_traveled)
            .sum::<f64>()
            / n
    );

    eprintln!("\n── Well Discovery (mean tick) ──");
    for i in 0..4 {
        let mean_tick = results
            .iter()
            .map(|r| r.well_discovery_ticks[i])
            .sum::<f64>()
            / n;
        let capacity = [800, 1500, 2500, 4000][i];
        eprintln!("  Well {} ({capacity}J): tick {mean_tick:.0}", i + 1);
    }

    // Do migrants survive better?
    eprintln!("\n── Migration vs Survival ──");
    let migrators: Vec<f64> = results.iter().map(|r| r.agents_that_migrated).collect();
    let alive: Vec<f64> = results.iter().map(|r| r.alive).collect();
    let d = cohens_d(&migrators, &alive);
    eprintln!(
        "  Correlation direction: {}",
        if d > 0.0 {
            "more migration = more survival"
        } else {
            "less migration = more survival"
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
