// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gradient Topology — how do structural changes modify the FEP landscape?
//!
//! The falsification experiment (F38) showed cooperation persists under
//! extreme PARAMETER changes. This experiment tests STRUCTURAL changes
//! that modify the topology of the gradient field itself:
//!
//! 1. BASELINE: Standard FEP gradient (full perception)
//! 2. SHORT_SIGHT: Well perception limited to 50 units (vs 200 default)
//! 3. FRAGMENTED: Well perception 30 units + harmony range 15 units
//! 4. OBSTACLES: 4 repulsive zones that push agents away (gradient barriers)
//! 5. MOVING_WELLS: Wells relocate every 2000 ticks (unstable attractors)
//!
//! The hypothesis: PARAMETER changes can't kill cooperation because the
//! gradient topology remains connected. STRUCTURAL changes that disconnect
//! the topology (fragmented perception, obstacles) should break it.
//!
//! Run: cargo run --example gradient_topology --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
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
enum Topology {
    Baseline,
    ShortSight,
    Fragmented,
    Obstacles,
    MovingWells,
}

struct TopoResult {
    condition: &'static str,
    alive: f64,
    clustering: f64,
    cooperation: f64,
    num_clusters: f64,
}

fn run_experiment(topo: Topology, seed: u64) -> TopoResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    if matches!(topo, Topology::Fragmented) {
        constants.harmony_range = 15.0; // very short social range
    }
    consciousness.constants = constants;

    let well_sight = match topo {
        Topology::ShortSight => 50.0,
        Topology::Fragmented => 30.0,
        _ => 200.0,
    };

    let mut wells = vec![
        SVector::from([50.0, 20.0]),
        SVector::from([-40.0, -30.0]),
        SVector::from([10.0, -50.0]),
    ];
    let mut well_remaining = vec![2500.0f64; 3];

    // Obstacles: repulsive zones at 4 points
    let obstacles: Vec<SVector<f64, 2>> = if matches!(topo, Topology::Obstacles) {
        vec![
            SVector::from([20.0, 20.0]),
            SVector::from([-20.0, 20.0]),
            SVector::from([20.0, -20.0]),
            SVector::from([-20.0, -20.0]),
        ]
    } else {
        vec![]
    };

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 120.0;
        let y = (rng_f64(&mut rng) - 0.5) * 120.0;
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

    let cond_name = match topo {
        Topology::Baseline => "BASELINE",
        Topology::ShortSight => "SHORT_SIGHT",
        Topology::Fragmented => "FRAGMENTED",
        Topology::Obstacles => "OBSTACLES",
        Topology::MovingWells => "MOVING_WELLS",
    };

    let mut coop = 0u64;

    for tick in 0..TICKS {
        // Moving wells: relocate every 2000 ticks
        if matches!(topo, Topology::MovingWells) && tick > 0 && tick % 2000 == 0 {
            for w in &mut wells {
                w[0] = (rng_f64(&mut rng) - 0.5) * 100.0;
                w[1] = (rng_f64(&mut rng) - 0.5) * 100.0;
            }
            // Refill wells partially on move
            for r in &mut well_remaining {
                *r = (*r + 500.0).min(2500.0);
            }
        }

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

        for &h in &handles {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position();

            // Well data limited by perception range
            let wdata: Vec<_> = wells
                .iter()
                .zip(well_remaining.iter())
                .filter(|&(ref w, &r)| r > 0.0 && (pos - *w).norm() < well_sight)
                .map(|(&p, &r)| (p, (r / 2500.0).min(1.0)))
                .collect();

            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();

            // Compute obstacle repulsion as danger source (strongest nearby obstacle)
            let danger = if !obstacles.is_empty() {
                obstacles
                    .iter()
                    .filter(|o| (pos - *o).norm() < 30.0)
                    .min_by(|a, b| (pos - *a).norm().partial_cmp(&(pos - *b).norm()).unwrap())
            } else {
                None
            };

            let danger_level = danger
                .map(|d| {
                    let dist = (pos - d).norm();
                    if dist < 30.0 {
                        (30.0 - dist) / 30.0
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);

            let dir = fep_gradient::free_energy_gradient(
                &pos,
                e.energy.fraction_remaining(),
                &e.harmony_activations,
                &near,
                &wdata,
                danger.map(|d| d as &SVector<f64, 2>),
                danger_level,
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

    let alive_handles: Vec<_> = handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    let alive = alive_handles.len() as f64;

    let positions: Vec<(usize, SVector<f64, 2>)> = alive_handles
        .iter()
        .enumerate()
        .filter_map(|(i, &h)| world.body(h).map(|b| (i, b.position())))
        .collect();
    let clusters = find_clusters(&positions, consciousness.constants.harmony_range);

    let clustering = if positions.len() >= 2 {
        positions
            .iter()
            .map(|(i, p)| {
                positions
                    .iter()
                    .filter(|(j, _)| j != i)
                    .map(|(_, q)| (p - q).norm())
                    .fold(f64::MAX, f64::min)
            })
            .sum::<f64>()
            / positions.len() as f64
    } else {
        f64::MAX
    };

    TopoResult {
        condition: cond_name,
        alive,
        clustering,
        cooperation: coop as f64,
        num_clusters: clusters.len() as f64,
    }
}

fn find_clusters(positions: &[(usize, SVector<f64, 2>)], range: f64) -> Vec<Vec<usize>> {
    let n = positions.len();
    if n == 0 {
        return vec![];
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        if p[i] != i {
            p[i] = find(p, p[i]);
        }
        p[i]
    }
    fn union(p: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(p, a), find(p, b));
        if ra != rb {
            p[ra] = rb;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if (positions[i].1 - positions[j].1).norm() < range {
                union(&mut parent, i, j);
            }
        }
    }
    let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        clusters.entry(root).or_default().push(i);
    }
    clusters.into_values().collect()
}

fn main() {
    eprintln!("=== Gradient Topology Experiment ===");
    eprintln!("How do structural changes modify the FEP landscape?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    println!("condition,seed,alive,clustering,cooperation,num_clusters");

    let topos = [
        (Topology::Baseline, "BASELINE"),
        (Topology::ShortSight, "SHORT_SIGHT"),
        (Topology::Fragmented, "FRAGMENTED"),
        (Topology::Obstacles, "OBSTACLES"),
        (Topology::MovingWells, "MOVING_WELLS"),
    ];
    let mut all: Vec<(&str, Vec<TopoResult>)> = Vec::new();

    for &(topo, name) in &topos {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(topo, seed);
            println!(
                "{},{seed},{:.1},{:.2},{:.0},{:.0}",
                r.condition, r.alive, r.clustering, r.cooperation, r.num_clusters
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, clusters={:.1}, coop={:.0}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.num_clusters).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n
        );
        all.push((name, results));
    }

    eprintln!("\n── Topology Effects ──");
    eprintln!("  Condition      Alive  Clusters  Clustering  Cooperation");
    for (name, results) in &all {
        let n = results.len() as f64;
        eprintln!(
            "  {:13} {:5.1}    {:5.1}     {:6.2}      {:9.0}",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.num_clusters).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n
        );
    }

    // Statistical tests vs baseline
    let baseline = &all[0].1;
    let b_alive: Vec<f64> = baseline.iter().map(|r| r.alive).collect();

    let mut p_tests = Vec::new();
    let mut d_tests = Vec::new();
    for (name, results) in &all[1..] {
        let n_alive: Vec<f64> = results.iter().map(|r| r.alive).collect();
        let (_, _, p) = mann_whitney_u(&b_alive, &n_alive);
        let d = cohens_d(&b_alive, &n_alive);
        p_tests.push((*name, p));
        d_tests.push(d);
    }

    let corrected = holm_bonferroni(&p_tests, 0.05);
    eprintln!("\n── vs BASELINE (Holm-Bonferroni, k=4) ──");
    for (i, &(label, adj_p, sig)) in corrected.iter().enumerate() {
        let d = d_tests[i];
        let size = if d.abs() > 0.8 {
            "LARGE"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        };
        eprintln!(
            "  vs {:13}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
            label,
            if sig { "← SIG" } else { "" }
        );
    }

    // Key question: did we break cooperation?
    let fragmented = &all[2].1;
    let frag_alive = fragmented.iter().map(|r| r.alive).sum::<f64>() / fragmented.len() as f64;
    let frag_coop = fragmented.iter().map(|r| r.cooperation).sum::<f64>() / fragmented.len() as f64;

    eprintln!("\n── Verdict ──");
    if frag_alive < 5.0 && frag_coop < 50000.0 {
        eprintln!(
            "  TOPOLOGY BREAKS COOPERATION: Fragmented perception destroys social structure."
        );
        eprintln!("  Parameters couldn't do it (F38), but topology can.");
    } else {
        eprintln!("  Cooperation persists under structural modification.");
        eprintln!("  Fragmented: {frag_alive:.1}/20 alive, {frag_coop:.0} cooperation events.");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
