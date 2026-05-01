// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hidden Resources — does the FEP gradient become indispensable when
//! resources are invisible?
//!
//! F46-48 showed the FEP gradient's social component HURTS when wells are
//! visible and static. But in ecology, social foraging provides critical
//! information benefits when resources are unpredictable (producer-scrounger
//! games, Giraldeau & Caraco 2000).
//!
//! When wells are HIDDEN, WELL_ONLY agents can't navigate to them (they
//! can't see what they can't perceive). The FEP gradient's social component
//! becomes an information channel: agents near wells attract other agents
//! via the resonance gradient, creating a social navigation network.
//!
//! 4 visibility × 3 controllers × 20 seeds
//!
//! Run: cargo run --example hidden_resources --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy)]
enum Visibility {
    Full,
    ShortRange,
    Hidden,
    SocialInfo,
}
#[derive(Clone, Copy)]
enum Ctrl {
    Fep,
    WellOnly,
    Greedy,
}

fn run_experiment(vis: Visibility, ctrl: Ctrl, seed: u64) -> (f64, f64, f64) {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    // Wells placed at moderate distance — findable if visible, challenging if hidden
    let wells = vec![
        SVector::from([50.0, 20.0]),
        SVector::from([-40.0, -30.0]),
        SVector::from([15.0, -45.0]),
    ];
    let mut well_remaining = vec![2000.0f64; 3];
    let mut rng = seed;
    let mut handles = Vec::new();

    // Start agents clustered near center — must FIND the wells
    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 40.0;
        let y = (rng_f64(&mut rng) - 0.5) * 40.0;
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

    let well_perception = match vis {
        Visibility::Full => 200.0,
        Visibility::ShortRange => 30.0,
        Visibility::Hidden => 0.0,     // can't see wells at all
        Visibility::SocialInfo => 0.0, // can't see wells directly
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

        // SOCIAL_INFO: agents near wells appear as "high energy" to the FEP gradient
        // This simulates information leakage — you can't see the well, but you can
        // see that your neighbor is healthy (high energy fraction)
        let agent_energy_info: Vec<(SVector<f64, 2>, f64)> =
            if matches!(vis, Visibility::SocialInfo) {
                handles
                    .iter()
                    .filter_map(|&h| {
                        let pos = world.body(h)?.position().0;
                        let ef = consciousness.entities.get(&h)?.energy.fraction_remaining();
                        Some((pos, ef))
                    })
                    .collect()
            } else {
                vec![]
            };

        for (idx, &h) in handles.iter().enumerate() {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position().0;

            // Wells visible only within perception range
            let wdata: Vec<_> = wells
                .iter()
                .zip(well_remaining.iter())
                .filter(|(w, &r)| r > 0.0 && (pos - *w).norm() < well_perception)
                .map(|(&p, &r)| (p, (r / 2000.0).min(1.0)))
                .collect();

            let vel: SVector<f64, 2> = match ctrl {
                Ctrl::Fep => {
                    let near: Vec<_> = adata
                        .iter()
                        .filter(|(p, _)| {
                            let d = (p - pos).norm();
                            d > 2.0 && d < consciousness.constants.harmony_range
                        })
                        .cloned()
                        .collect();

                    // SOCIAL_INFO: add attraction toward high-energy agents
                    // (they're probably near a well)
                    let mut social_wells: Vec<(SVector<f64, 2>, f64)> = wdata.clone();
                    if matches!(vis, Visibility::SocialInfo) {
                        for (apos, aef) in &agent_energy_info {
                            let d = (apos - pos).norm();
                            if d > 5.0 && d < consciousness.constants.harmony_range && *aef > 0.7 {
                                // Healthy agent = pseudo-well signal
                                social_wells.push((*apos, *aef * 0.5));
                            }
                        }
                    }

                    fep_gradient::free_energy_gradient(
                        &pos,
                        e.energy.fraction_remaining(),
                        &e.harmony_activations,
                        &near,
                        &social_wells,
                        None,
                        0.0,
                    ) * 20.0
                }
                Ctrl::WellOnly => {
                    // Can only navigate to VISIBLE wells
                    let mut best_dir = SVector::zeros();
                    let mut best_dist = f64::MAX;
                    for (wp, wr) in &wdata {
                        if *wr < 0.01 {
                            continue;
                        }
                        let delta = wp - pos;
                        let d = delta.norm();
                        if d < best_dist && d > 1.0 {
                            best_dist = d;
                            best_dir = delta / d;
                        }
                    }
                    if best_dist == f64::MAX {
                        // No visible well — explore randomly
                        let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                        SVector::from([angle.cos() * 15.0, angle.sin() * 15.0])
                    } else {
                        best_dir * 20.0
                    }
                }
                Ctrl::Greedy => {
                    let mut best_dir = SVector::zeros();
                    let mut best_gain = f64::NEG_INFINITY;
                    for ai in 0..8 {
                        let angle = ai as f64 * std::f64::consts::TAU / 8.0;
                        let td = SVector::from([angle.cos(), angle.sin()]);
                        let tp = pos + td * 5.0;
                        let mut gain = 0.0;
                        for (wp, wr) in &wdata {
                            if *wr < 0.01 {
                                continue;
                            }
                            if (tp - wp).norm() < 35.0 {
                                gain += consciousness.constants.energy_well_regen_rate * wr;
                            }
                        }
                        if gain > best_gain {
                            best_gain = gain;
                            best_dir = td;
                        }
                    }
                    if best_gain <= 0.0 {
                        let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                        SVector::from([angle.cos() * 15.0, angle.sin() * 15.0])
                    } else {
                        best_dir * 20.0
                    }
                }
            };
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = vel;
            }
        }

        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr_rate = consciousness.constants.energy_well_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                // Wells regenerate regardless of visibility (you benefit if you're on one)
                if let Some(b) = world.body(h) {
                    let pos = b.position().0;
                    for (wi, &w) in wells.iter().enumerate() {
                        if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                            let d = wr_rate.min(well_remaining[wi]);
                            e.energy.regenerate(d);
                            well_remaining[wi] -= d;
                            break;
                        }
                    }
                }
            }
        }

        // Resonance always active (passive cooperation)
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
    (alive, energy, coop as f64)
}

fn main() {
    eprintln!("=== Hidden Resources ===");
    eprintln!("Does the FEP gradient become indispensable when resources are invisible?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    println!("visibility,controller,seed,alive,energy,cooperation");

    let viss = [
        (Visibility::Full, "FULL"),
        (Visibility::ShortRange, "SHORT"),
        (Visibility::Hidden, "HIDDEN"),
        (Visibility::SocialInfo, "SOCIAL"),
    ];
    let ctrls = [
        (Ctrl::WellOnly, "WELL"),
        (Ctrl::Fep, "FEP"),
        (Ctrl::Greedy, "GREEDY"),
    ];

    let mut all: Vec<(&str, &str, Vec<f64>)> = Vec::new();

    for &(vis, vname) in &viss {
        for &(ctrl, cname) in &ctrls {
            let mut alive_vec = Vec::new();
            for s in 0..SEEDS {
                let seed = 42 + s as u64 * 997;
                eprintln!("  {vname}+{cname} seed={seed}...");
                let (alive, energy, coop) = run_experiment(vis, ctrl, seed);
                println!("{vname},{cname},{seed},{alive:.1},{energy:.1},{coop:.0}");
                alive_vec.push(alive);
            }
            let mean = alive_vec.iter().sum::<f64>() / SEEDS as f64;
            eprintln!("  → {vname}+{cname}: alive={mean:.1}");
            all.push((vname, cname, alive_vec));
        }
    }

    // Summary table
    eprintln!("\n── Visibility × Controller ──");
    eprintln!("  {:8}  {:6}  {:6}  {:6}", "", "WELL", "FEP", "GREEDY");
    for vname in &["FULL", "SHORT", "HIDDEN", "SOCIAL"] {
        let w = all
            .iter()
            .find(|(v, c, _)| v == vname && *c == "WELL")
            .map(|(_, _, a)| a.iter().sum::<f64>() / a.len() as f64)
            .unwrap_or(0.0);
        let f = all
            .iter()
            .find(|(v, c, _)| v == vname && *c == "FEP")
            .map(|(_, _, a)| a.iter().sum::<f64>() / a.len() as f64)
            .unwrap_or(0.0);
        let g = all
            .iter()
            .find(|(v, c, _)| v == vname && *c == "GREEDY")
            .map(|(_, _, a)| a.iter().sum::<f64>() / a.len() as f64)
            .unwrap_or(0.0);
        let best = if w >= f && w >= g {
            "WELL"
        } else if f >= g {
            "FEP"
        } else {
            "GREEDY"
        };
        eprintln!(
            "  {:8}  {:5.1}   {:5.1}   {:5.1}   ← {best} wins",
            vname, w, f, g
        );
    }

    // THE KEY TEST: Under HIDDEN, does FEP outperform WELL?
    let hidden_fep = all
        .iter()
        .find(|(v, c, _)| *v == "HIDDEN" && *c == "FEP")
        .unwrap();
    let hidden_well = all
        .iter()
        .find(|(v, c, _)| *v == "HIDDEN" && *c == "WELL")
        .unwrap();
    let d = cohens_d(&hidden_well.2, &hidden_fep.2);
    let (_, _, p) = mann_whitney_u(&hidden_well.2, &hidden_fep.2);

    eprintln!("\n── KEY TEST: HIDDEN resources — FEP vs WELL ──");
    eprintln!(
        "  FEP={:.1}, WELL={:.1}, d={d:.3}, p={p:.4}",
        hidden_fep.2.iter().sum::<f64>() / hidden_fep.2.len() as f64,
        hidden_well.2.iter().sum::<f64>() / hidden_well.2.len() as f64
    );
    if d < -0.5 {
        eprintln!("  FEP RECOVERS: Social gradient is indispensable when resources are hidden.");
        eprintln!("  The social drive is an information channel, not a metabolic cost.");
    } else if d > 0.5 {
        eprintln!("  WELL STILL WINS: Even random exploration outperforms social seeking.");
    } else {
        eprintln!("  NO DIFFERENCE: Both controllers perform similarly under hidden resources.");
    }

    // SOCIAL_INFO test
    let social_fep = all
        .iter()
        .find(|(v, c, _)| *v == "SOCIAL" && *c == "FEP")
        .unwrap();
    let social_well = all
        .iter()
        .find(|(v, c, _)| *v == "SOCIAL" && *c == "WELL")
        .unwrap();
    let d2 = cohens_d(&social_well.2, &social_fep.2);
    eprintln!("\n── SOCIAL_INFO: FEP with energy-signal wells ──");
    eprintln!(
        "  FEP={:.1}, WELL={:.1}, d={d2:.3}",
        social_fep.2.iter().sum::<f64>() / social_fep.2.len() as f64,
        social_well.2.iter().sum::<f64>() / social_well.2.len() as f64
    );

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
