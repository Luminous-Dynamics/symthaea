// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Large-Scale Cooperation — does cooperation emerge at N=1000?
//!
//! The definitive scale test. Runs the core cooperation experiment with
//! spatial hashing at N=1000 agents. Measures whether Dunbar-like cluster
//! limits hold at scale, and whether the FEP gradient still produces
//! meaningful social structure when the arena is crowded.
//!
//! Run: cargo run --example cooperation_1000 --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::cohens_d;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::spatial_hash::SpatialHash;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const TICKS: usize = 4_000; // shorter due to scale
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 5; // fewer seeds due to wall-clock cost
const ARENA: f64 = 400.0;

const POPULATIONS: [usize; 4] = [100, 250, 500, 1000];

const HARMONY_PROFILES: [[f64; 9]; 6] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
    [0.6, 0.2, 0.4, 0.3, 0.5, 0.4, 0.2, 0.5, 0.5],
    [0.3, 0.5, 0.2, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5],
];

struct ScaleResult {
    n: usize,
    alive: f64,
    survival_rate: f64,
    cooperation: f64,
    coop_per_agent: f64,
    num_clusters: f64,
    mean_cluster_size: f64,
    wall_ms: f64,
}

fn run_experiment(n_agents: usize, seed: u64) -> ScaleResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.20; // lighter pressure for large N
    consciousness.constants = constants;

    // Wells scale with population: 4 base + 1 per 100 agents
    let n_wells = 4 + n_agents / 100;
    let mut wells = Vec::new();
    let mut well_remaining = Vec::new();
    let mut rng = seed;
    for _ in 0..n_wells {
        let x = (rng_f64(&mut rng) - 0.5) * ARENA * 0.8;
        let y = (rng_f64(&mut rng) - 0.5) * ARENA * 0.8;
        wells.push(SVector::from([x, y]));
        well_remaining.push(2000.0f64);
    }

    let mut handles = Vec::new();
    for i in 0..n_agents {
        let x = (rng_f64(&mut rng) - 0.5) * ARENA;
        let y = (rng_f64(&mut rng) - 0.5) * ARENA;
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
            e.harmony_activations = HARMONY_PROFILES[i % 6];
        }
        handles.push(h);
    }

    let mut spatial = SpatialHash::<2>::new(consciousness.constants.harmony_range);
    let mut coop = 0u64;

    let start = std::time::Instant::now();

    for _tick in 0..TICKS {
        // Consciousness update (batch)
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

        // Build spatial hash
        spatial.clear();
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(b) = world.body(h) {
                spatial.insert(idx, &b.position().0);
            }
        }

        // FEP gradient with spatial hash
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
            .map(|(&p, &r)| (p, (r / 2000.0).min(1.0)))
            .collect();

        for (idx, &h) in handles.iter().enumerate() {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position().0;

            let near: Vec<_> = spatial
                .query_neighbors(&pos)
                .into_iter()
                .filter(|&j| j != idx)
                .filter_map(|j| {
                    if j < adata.len() {
                        let (p, h) = &adata[j];
                        let d = (p - pos).norm();
                        if d > 2.0 && d < consciousness.constants.harmony_range {
                            Some((p.clone(), h.clone()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
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

        // Maintenance
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

        // Cooperation via spatial hash
        for (idx, &h) in handles.iter().enumerate() {
            let Some(e_a) = consciousness.entities.get(&h) else {
                continue;
            };
            let harm_a = e_a.harmony_activations;
            let pos_a = match world.body(h) {
                Some(b) => b.position().0,
                None => continue,
            };

            for &j in &spatial.query_neighbors(&pos_a) {
                if j <= idx {
                    continue;
                }
                let hb = handles[j];
                let in_range = match world.body(hb) {
                    Some(b) => {
                        (b.position().0 - pos_a).norm() < consciousness.constants.harmony_range
                    }
                    None => false,
                };
                if !in_range {
                    continue;
                }
                let harm_b = match consciousness.entities.get(&hb) {
                    Some(e) => e.harmony_activations,
                    None => continue,
                };
                let res = HarmonyField::<2>::resonance(&harm_a, &harm_b);
                if res > 0.5 {
                    let rg =
                        consciousness.constants.harmony_resonance_regen_rate * (res - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&h) {
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

    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

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

    // Cluster detection
    let positions: Vec<(usize, SVector<f64, 2>)> = alive_handles
        .iter()
        .enumerate()
        .filter_map(|(i, h)| world.body(*h).map(|b| (i, b.position().0)))
        .collect();
    let clusters = find_clusters(&positions, consciousness.constants.harmony_range);

    ScaleResult {
        n: n_agents,
        alive,
        survival_rate: alive / n_agents as f64,
        cooperation: coop as f64,
        coop_per_agent: coop as f64 / n_agents as f64,
        num_clusters: clusters.len() as f64,
        mean_cluster_size: if clusters.is_empty() {
            0.0
        } else {
            clusters.iter().map(|c| c.len() as f64).sum::<f64>() / clusters.len() as f64
        },
        wall_ms,
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

    // Use spatial hash for cluster detection too
    let mut hash = SpatialHash::<2>::new(range);
    for &(i, ref pos) in positions {
        hash.insert(i, pos);
    }

    for (local_i, &(_, ref pos_a)) in positions.iter().enumerate() {
        for &j in &hash.query_neighbors(pos_a) {
            if j <= local_i {
                continue;
            }
            if j < positions.len() && (positions[j].1 - pos_a).norm() < range {
                union(&mut parent, local_i, j);
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
    eprintln!("=== Large-Scale Cooperation ===");
    eprintln!("Does cooperation emerge at N=1000?");
    eprintln!("Arena: {ARENA}×{ARENA}, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("Using spatial hash for O(n×k) neighbor queries");

    println!("n,seed,alive,survival,cooperation,coop_per_agent,clusters,mean_cluster,wall_ms");

    let mut all_results: Vec<(usize, Vec<ScaleResult>)> = Vec::new();

    for &n in &POPULATIONS {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  N={n} seed={seed}...");
            let r = run_experiment(n, seed);
            println!(
                "{},{seed},{:.0},{:.3},{:.0},{:.0},{:.0},{:.1},{:.0}",
                r.n,
                r.alive,
                r.survival_rate,
                r.cooperation,
                r.coop_per_agent,
                r.num_clusters,
                r.mean_cluster_size,
                r.wall_ms
            );
            eprintln!(
                "    {:.0}% alive, {:.0} clusters (mean size {:.1}), {:.0}s",
                r.survival_rate * 100.0,
                r.num_clusters,
                r.mean_cluster_size,
                r.wall_ms / 1000.0
            );
            results.push(r);
        }
        all_results.push((n, results));
    }

    eprintln!("\n── Population Scaling at Scale ��─");
    eprintln!("  N      Survival  Clusters  MeanSize  Coop/Agent  Wall(s)");
    for (n, results) in &all_results {
        let ns = results.len() as f64;
        eprintln!(
            "  {:5}  {:5.0}%    {:5.1}     {:6.1}    {:8.0}    {:5.0}s",
            n,
            results.iter().map(|r| r.survival_rate).sum::<f64>() / ns * 100.0,
            results.iter().map(|r| r.num_clusters).sum::<f64>() / ns,
            results.iter().map(|r| r.mean_cluster_size).sum::<f64>() / ns,
            results.iter().map(|r| r.coop_per_agent).sum::<f64>() / ns,
            results.iter().map(|r| r.wall_ms).sum::<f64>() / ns / 1000.0
        );
    }

    // Does cluster size plateau (Dunbar)?
    if all_results.len() >= 2 {
        let small = &all_results[0].1;
        let large = &all_results.last().unwrap().1;
        let s_clust: Vec<f64> = small.iter().map(|r| r.mean_cluster_size).collect();
        let l_clust: Vec<f64> = large.iter().map(|r| r.mean_cluster_size).collect();
        let d = cohens_d(&s_clust, &l_clust);
        eprintln!("\n── Cluster size: N=100 vs N=1000 ──");
        eprintln!(
            "  Cohen's d = {d:.3} ({})",
            if d.abs() > 0.8 {
                "large — scales with N"
            } else {
                "small — Dunbar limit holds"
            }
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
