// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dunbar Number — does thermodynamic pressure produce natural group size limits?
//!
//! Sweeps population from 4 to 128 agents in a fixed-size arena. Measures
//! how many distinct clusters form, mean cluster size, and whether survival
//! per capita has a sweet spot (Dunbar's number for this physics).
//!
//! Dunbar (1992) found primate group size correlates with neocortex ratio.
//! Our analogue: group size should correlate with harmony range (social reach)
//! and maintenance cost (cognitive overhead of integration).
//!
//! Run: cargo run --example dunbar_number --release

use nalgebra::SVector;
use symtropy_consciousness_physics::convergence::cohens_d;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;
use symthaea_consciousness_equation::ConsciousnessInputs;

const TICKS: usize = 6_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 6;
const ARENA_SIZE: f64 = 200.0; // fixed arena regardless of N

const POPULATIONS: [usize; 8] = [4, 8, 12, 20, 32, 48, 64, 96];

const HARMONY_PROFILES: [[f64; 8]; 6] = [
    [0.8, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.7],
    [0.7, 0.4, 0.3, 0.1, 0.3, 0.2, 0.3, 0.6],
    [0.1, 0.2, 0.7, 0.3, 0.8, 0.2, 0.1, 0.2],
    [0.2, 0.1, 0.6, 0.4, 0.7, 0.3, 0.2, 0.1],
    [0.3, 0.6, 0.2, 0.7, 0.2, 0.6, 0.3, 0.3],
    [0.4, 0.2, 0.3, 0.6, 0.3, 0.7, 0.4, 0.2],
];

struct DunbarResult {
    n_agents: usize,
    alive: f64,
    survival_rate: f64,
    num_clusters: f64,
    mean_cluster_size: f64,
    max_cluster_size: f64,
    per_capita_energy: f64,
    cooperation_per_agent: f64,
}

fn run_experiment(n_agents: usize, seed: u64) -> DunbarResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    consciousness.constants = constants;

    // Wells scale with arena, not population — creates density pressure
    let wells = vec![
        SVector::from([50.0, 0.0]),
        SVector::from([-50.0, 0.0]),
        SVector::from([0.0, 50.0]),
        SVector::from([0.0, -50.0]),
    ];
    let mut well_remaining = vec![1500.0f64; 4];

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..n_agents {
        let x = (rng_f64(&mut rng) - 0.5) * ARENA_SIZE;
        let y = (rng_f64(&mut rng) - 0.5) * ARENA_SIZE;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) { b.linear_damping = 0.05; }
        consciousness.register(h, consciousness.constants.initial_energy, consciousness.constants.harmony_range);
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = HARMONY_PROFILES[i % 6];
        }
        handles.push(h);
    }

    let mut cooperation_events = 0u64;

    for _tick in 0..TICKS {
        for &h in &handles {
            let e = consciousness.entities.get(&h);
            let ef = e.map(|e| e.energy.fraction_remaining()).unwrap_or(0.0);
            let pe = e.map(|e| e.prediction_error).unwrap_or(0.0);
            let ht = e.map(|e| e.total_harmony_energy()).unwrap_or(0.0);
            let collapsed = e.map(|e| e.energy.is_collapsed()).unwrap_or(true);
            let inputs = if collapsed {
                ConsciousnessInputs { phi: 0.0, broadcast: 0.0, working_memory: 0.0,
                    attention: 0.0, recurrence: 0.0, embodiment: 0.0, knowledge: 0.0, synchrony: 0.0 }
            } else {
                ConsciousnessInputs {
                    phi: ef, broadcast: 0.5, working_memory: (1.0 - pe).max(0.0),
                    attention: 0.5, recurrence: 1.0, embodiment: 0.7,
                    knowledge: (ht / 8.0).min(1.0), synchrony: consciousness.collective_phi.max(0.5),
                }
            };
            consciousness.update_entity(h, &inputs, Point::origin());
        }

        let adata: Vec<_> = handles.iter().filter_map(|&h| {
            Some((world.body(h)?.position().0, consciousness.entities.get(&h)?.harmony_activations))
        }).collect();
        let wdata: Vec<_> = wells.iter().zip(well_remaining.iter())
            .filter(|(_, &r)| r > 0.0).map(|(&p, &r)| (p, (r / 1500.0).min(1.0))).collect();
        for &h in &handles {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else { continue };
            if e.energy.is_collapsed() { continue; }
            let pos = b.position().0;
            let near: Vec<_> = adata.iter().filter(|(p, _)| {
                let d = (p - pos).norm(); d > 2.0 && d < consciousness.constants.harmony_range
            }).cloned().collect();
            let dir = fep_gradient::free_energy_gradient(&pos, e.energy.fraction_remaining(),
                &e.harmony_activations, &near, &wdata, None, 0.0);
            if let Some(b) = world.body_mut(h) { b.linear_velocity = dir * 20.0; }
        }

        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let nw: Vec<Option<usize>> = handles.iter().map(|&h| {
            world.body(h).and_then(|b| {
                let pos = b.position().0;
                wells.iter().enumerate().find(|(i, &w)| (pos - w).norm() < 35.0 && well_remaining[*i] > 0.0).map(|(i,_)| i)
            })
        }).collect();
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) { e.energy.tick_reset(); }
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

        for i in 0..handles.len() {
            for j in (i+1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let in_range = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => a.position().distance(b.position()) < consciousness.constants.harmony_range,
                    _ => false,
                };
                if !in_range { continue; }
                let (a, b) = match (consciousness.entities.get(&ha), consciousness.entities.get(&hb)) {
                    (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations), _ => continue,
                };
                let res = HarmonyField::<2>::resonance(&a, &b);
                if res > 0.5 {
                    let rg = consciousness.constants.harmony_resonance_regen_rate * (res - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) { e.energy.regenerate(rg); }
                    if let Some(e) = consciousness.entities.get_mut(&hb) { e.energy.regenerate(rg); }
                    cooperation_events += 1;
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    let alive_handles: Vec<_> = handles.iter().filter(|h| {
        consciousness.entities.get(h).map(|e| !e.energy.is_collapsed()).unwrap_or(false)
    }).cloned().collect();
    let alive = alive_handles.len() as f64;
    let total_energy: f64 = handles.iter().filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available)).sum();

    // Cluster detection: connected components within harmony_range
    let positions: Vec<(usize, SVector<f64, 2>)> = alive_handles.iter().enumerate()
        .filter_map(|(i, h)| world.body(*h).map(|b| (i, b.position().0))).collect();
    let clusters = find_clusters(&positions, consciousness.constants.harmony_range);

    DunbarResult {
        n_agents,
        alive,
        survival_rate: alive / n_agents as f64,
        num_clusters: clusters.len() as f64,
        mean_cluster_size: if clusters.is_empty() { 0.0 } else { clusters.iter().map(|c| c.len() as f64).sum::<f64>() / clusters.len() as f64 },
        max_cluster_size: clusters.iter().map(|c| c.len() as f64).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0),
        per_capita_energy: if alive > 0.0 { total_energy / alive } else { 0.0 },
        cooperation_per_agent: cooperation_events as f64 / n_agents as f64,
    }
}

fn find_clusters(positions: &[(usize, SVector<f64, 2>)], range: f64) -> Vec<Vec<usize>> {
    let n = positions.len();
    if n == 0 { return vec![]; }
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i { parent[i] = find(parent, parent[i]); }
        parent[i]
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb { parent[ra] = rb; }
    }

    for i in 0..n {
        for j in (i+1)..n {
            if (positions[i].1 - positions[j].1).norm() < range {
                union(&mut parent, i, j);
            }
        }
    }

    let mut clusters: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        clusters.entry(root).or_default().push(i);
    }
    clusters.into_values().collect()
}

fn main() {
    eprintln!("=== Dunbar Number Experiment ===");
    eprintln!("Does thermodynamic pressure produce natural group size limits?");
    eprintln!("Arena: {ARENA_SIZE}x{ARENA_SIZE}, {TICKS} ticks, {SEEDS} seeds");

    println!("n_agents,seed,alive,survival_rate,num_clusters,mean_cluster,max_cluster,per_capita_energy,coop_per_agent");

    let mut all_results: Vec<(usize, Vec<DunbarResult>)> = Vec::new();

    for &n in &POPULATIONS {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  N={n} seed={seed}...");
            let r = run_experiment(n, seed);
            println!("{},{seed},{:.1},{:.3},{:.1},{:.1},{:.0},{:.1},{:.0}",
                r.n_agents, r.alive, r.survival_rate, r.num_clusters,
                r.mean_cluster_size, r.max_cluster_size, r.per_capita_energy, r.cooperation_per_agent);
            results.push(r);
        }
        let ns = results.len() as f64;
        let surv = results.iter().map(|r| r.survival_rate).sum::<f64>() / ns;
        let clust = results.iter().map(|r| r.mean_cluster_size).sum::<f64>() / ns;
        let ncl = results.iter().map(|r| r.num_clusters).sum::<f64>() / ns;
        eprintln!("  → N={n}: survival={surv:.1}%, clusters={ncl:.1}, mean_size={clust:.1}");
        all_results.push((n, results));
    }

    eprintln!("\n── Population Scaling ──");
    eprintln!("  N    Survival  Clusters  MeanSize  MaxSize  Energy/cap  Coop/agent");
    for (n, results) in &all_results {
        let ns = results.len() as f64;
        let surv = results.iter().map(|r| r.survival_rate).sum::<f64>() / ns;
        let ncl = results.iter().map(|r| r.num_clusters).sum::<f64>() / ns;
        let mcs = results.iter().map(|r| r.mean_cluster_size).sum::<f64>() / ns;
        let maxc = results.iter().map(|r| r.max_cluster_size).sum::<f64>() / ns;
        let ener = results.iter().map(|r| r.per_capita_energy).sum::<f64>() / ns;
        let coop = results.iter().map(|r| r.cooperation_per_agent).sum::<f64>() / ns;
        eprintln!("  {n:3}   {surv:5.1}%    {ncl:5.1}     {mcs:5.1}     {maxc:5.0}      {ener:5.1}J    {coop:7.0}");
    }

    // Per-capita energy scaling: log-log regression
    eprintln!("\n── Per-Capita Energy Scaling ──");
    let points: Vec<(f64, f64)> = all_results.iter().map(|(n, results)| {
        let ns = results.len() as f64;
        let ener = results.iter().map(|r| r.per_capita_energy).sum::<f64>() / ns;
        ((*n as f64).ln(), ener.max(0.01).ln())
    }).collect();
    if points.len() >= 2 {
        let (slope, intercept) = linear_regression(&points);
        eprintln!("  log(E/N) = {slope:.3} × log(N) + {intercept:.3}");
        eprintln!("  Scaling exponent beta = {slope:.3}");
        eprintln!("  (Hou et al. 2010 eusocial: beta = -0.19)");
    }

    // Optimal group size
    let best = all_results.iter().max_by(|(_, a), (_, b)| {
        let sa = a.iter().map(|r| r.survival_rate).sum::<f64>() / a.len() as f64;
        let sb = b.iter().map(|r| r.survival_rate).sum::<f64>() / b.len() as f64;
        sa.partial_cmp(&sb).unwrap()
    });
    if let Some((n, _)) = best {
        eprintln!("\n  Optimal population: N={n} (highest per-capita survival)");
    }

    eprintln!("\n=== Complete ===");
}

fn linear_regression(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|(x, _)| x).sum();
    let sy: f64 = points.iter().map(|(_, y)| y).sum();
    let sxy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sxx: f64 = points.iter().map(|(x, _)| x * x).sum();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

fn rng_f64(s: &mut u64) -> f64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (*s >> 11) as f64 / (1u64 << 53) as f64 }
