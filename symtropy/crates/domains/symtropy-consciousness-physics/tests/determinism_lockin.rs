// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Determinism lock-in test: verify bitwise reproducibility.

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 8;
const TICKS: usize = 500;
const DT: f64 = 1.0 / 64.0;

struct SimState {
    positions: Vec<[f64; 2]>,
    energies: Vec<f64>,
    cooperation: u64,
}

fn run_mini_sim(seed: u64) -> SimState {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([20.0, 0.0])];
    let mut well_remaining = vec![500.0f64];

    let mut rng = seed;
    let mut handles = Vec::new();
    let profiles = [
        [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.0],
        [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.0],
    ];

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 50.0;
        let y = (rng_f64(&mut rng) - 0.5) * 50.0;
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
            e.harmony_activations = profiles[i % 2];
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
            .map(|(&p, &r)| (p, (r / 500.0).min(1.0)))
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

        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar);
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
                        (a.position() - b.position()).norm() < consciousness.constants.harmony_range
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

    let positions: Vec<[f64; 2]> = handles
        .iter()
        .map(|&h| {
            world
                .body(h)
                .map(|b| {
                    let p = b.position();
                    [p[0], p[1]]
                })
                .unwrap_or([0.0, 0.0])
        })
        .collect();
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

    SimState {
        positions,
        energies,
        cooperation: coop,
    }
}

#[test]
fn determinism_lockin_seed_42() {
    let a = run_mini_sim(42);
    let b = run_mini_sim(42);

    assert_eq!(a.cooperation, b.cooperation, "Cooperation count diverged");
    for i in 0..AGENTS {
        assert_eq!(
            a.positions[i][0].to_bits(),
            b.positions[i][0].to_bits(),
            "Agent {i} x-position diverged: {} vs {}",
            a.positions[i][0],
            b.positions[i][0]
        );
        assert_eq!(
            a.positions[i][1].to_bits(),
            b.positions[i][1].to_bits(),
            "Agent {i} y-position diverged"
        );
        assert_eq!(
            a.energies[i].to_bits(),
            b.energies[i].to_bits(),
            "Agent {i} energy diverged: {} vs {}",
            a.energies[i],
            b.energies[i]
        );
    }
}

#[test]
fn different_seeds_different_results() {
    let a = run_mini_sim(42);
    let b = run_mini_sim(999);
    // At least positions should differ
    let any_diff = a
        .positions
        .iter()
        .zip(b.positions.iter())
        .any(|(pa, pb)| pa[0].to_bits() != pb[0].to_bits());
    assert!(any_diff, "Different seeds should produce different results");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
