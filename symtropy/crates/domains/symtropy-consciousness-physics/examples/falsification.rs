// SPDX-License-Identifier: AGPL-3.0-or-later

use nalgebra::SVector;
use rand::prelude::*;
use std::time::Instant;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::{ConsciousnessField, fep_gradient};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

struct FalsifyResult {
    #[allow(dead_code)]
    condition: &'static str,
    alive: usize,
    phi_avg: f64,
    energy_avg: f64,
    well_dist_avg: f64,
}

fn run_simulation(
    ticks: usize,
    agents: usize,
    wells_count: usize,
    maintenance: f64,
    range: f64,
) -> FalsifyResult {
    let mut rng = StdRng::seed_from_u64(42);
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::new();

    let mut wells = Vec::new();
    let well_cap: f64 = 5000.0;
    let mut well_remaining = vec![well_cap; wells_count];
    for _ in 0..wells_count {
        wells.push(SVector::from([
            rng.gen_range(-150.0..150.0),
            rng.gen_range(-150.0..150.0),
        ]));
    }

    let mut handles = Vec::new();
    for _ in 0..agents {
        let pos = Point::new([rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)]);
        let h = world.add_sphere(pos, 1.0, 1.0);
        consciousness.register(h, 100.0, 10.0);
        handles.push(h);
    }

    let ambient: f64 = 0.5;

    for _ in 0..ticks {
        world.step(0.1);

        let positions: Vec<_> = handles
            .iter()
            .filter_map(|&h| Some((h, world.body(h)?.transform.translation)))
            .collect();
        consciousness.rebuild_harmony_field(&positions);
        consciousness.tick_thermodynamics();

        for &h in &handles {
            let (pos, pe, ef, ht) = if let Some(b) = world.body(h) {
                let pos = b.position();
                let e = consciousness.entities.get(&h).unwrap();
                let near_wells: Vec<_> = wells
                    .iter()
                    .zip(well_remaining.iter())
                    .filter(|&(_, &r)| r > 0.0)
                    .filter(|&(w, _)| (*w - pos).norm() < 30.0)
                    .collect();

                let pe = if near_wells.is_empty() {
                    1.0_f64
                } else {
                    0.1_f64
                };
                (
                    pos,
                    pe,
                    e.energy.fraction_remaining(),
                    e.total_harmony_energy(),
                )
            } else {
                continue;
            };

            let inputs = if pe > 0.5 {
                ConsciousnessInputs {
                    phi: ef * 0.5,
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
            consciousness.update_entity(h, &inputs, Point::new([0.0, 0.0]));
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
            .map(|(&p, &r)| (p, (r / well_cap).min(1.0_f64)))
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
            if let Some(b_mut) = world.body_mut(h) {
                b_mut.linear_velocity = dir * 20.0;
            }
        }

        let rm = consciousness.resource_regeneration_multiplier();
        let wr: f64 = 0.5; // Default regen rate
        for &h in handles.iter() {
            consciousness.consume_energy(h, maintenance * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ambient * rm);
                if let Some(b) = world.body(h) {
                    let pos = b.position();
                    for (wi, &w) in wells.iter().enumerate() {
                        if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                            let d = wr.min(well_remaining[wi]);
                            e.energy.regenerate(d);
                            well_remaining[wi] -= d;
                        }
                    }
                }
            }
        }
    }

    let alive: Vec<_> = handles
        .iter()
        .filter(|&h| !consciousness.entities.get(h).unwrap().energy.is_collapsed())
        .collect();

    let phi_avg = if !alive.is_empty() {
        alive.iter().map(|&h| consciousness.phi(*h)).sum::<f64>() / alive.len() as f64
    } else {
        0.0
    };

    let energy_avg = if !alive.is_empty() {
        alive
            .iter()
            .map(|&h| consciousness.entities.get(h).unwrap().energy.available)
            .sum::<f64>()
            / alive.len() as f64
    } else {
        0.0
    };

    let well_dist_avg = if !alive.is_empty() {
        alive
            .iter()
            .map(|&h| {
                let pos = world.body(*h).unwrap().position();
                wells
                    .iter()
                    .map(|&q| (pos - q).norm())
                    .fold(f64::MAX, f64::min)
            })
            .sum::<f64>()
            / alive.len() as f64
    } else {
        f64::MAX
    };

    FalsifyResult {
        condition: "",
        alive: alive.len(),
        phi_avg,
        energy_avg,
        well_dist_avg,
    }
}

fn main() {
    println!("=== Consciousness Falsification Test ===");
    let start = Instant::now();

    let baseline = run_simulation(500, 50, 2, 5.0, 50.0);
    println!(
        "Baseline: Alive={}, Phi={:.4}, Energy={:.1}, Dist={:.1}",
        baseline.alive, baseline.phi_avg, baseline.energy_avg, baseline.well_dist_avg
    );

    let high_maintenance = run_simulation(500, 50, 2, 12.0, 50.0);
    println!(
        "High Maint: Alive={}, Phi={:.4}, Energy={:.1}, Dist={:.1}",
        high_maintenance.alive,
        high_maintenance.phi_avg,
        high_maintenance.energy_avg,
        high_maintenance.well_dist_avg
    );

    println!("\nSimulation completed in {:?}", start.elapsed());

    if high_maintenance.alive < baseline.alive {
        println!("PASSED: High maintenance led to more collapses.");
    } else {
        println!("FAILED: Maintenance impact not observed.");
    }
}
