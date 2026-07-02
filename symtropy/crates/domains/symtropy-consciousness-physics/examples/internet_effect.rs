// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Internet Effect Experiment — what has technology done to consciousness-cooperation?
//!
//! Models pre-internet vs post-internet social dynamics by varying:
//! - Harmony range (local → global reach)
//! - Maintenance cost (cognitive load → information overload)
//! - Resonance threshold (open cooperation → echo chambers)
//! - "Invisible influencers" (bots/algorithms that emit harmony without being agents)
//!
//! Maps to the dimensional sweep finding: internet collapsed 3D social space
//! into effectively 2D (everyone reachable), producing tighter clustering
//! (echo chambers) but faster energy depletion (burnout).
//!
//! Run: cargo run --example internet_effect --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 24;
const TICKS: usize = 5_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 10;

#[derive(Clone)]
struct Era {
    name: &'static str,
    harmony_range: f64,       // how far can you "feel" others
    maintenance: f64,         // cognitive cost per tick
    resonance_threshold: f64, // min resonance for cooperation
    influencer_count: usize,  // invisible field emitters (bots/algorithms)
    influencer_strength: f64, // how strong their field is
}

struct EraResult {
    alive: f64,
    energy: f64,
    clustering: f64,
    num_clusters: f64, // distinct groups (connected components at threshold)
    polarization: f64, // variance of harmony across agents
    cooperation_events: f64,
    burnout_rate: f64, // fraction of agents that collapsed at any point
}

fn run_era(era: &Era, seed: u64) -> EraResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.harmony_range = era.harmony_range;
    constants.consciousness_maintenance_per_tick = era.maintenance;
    consciousness.constants = constants;

    let mut rng = seed;
    let mut handles = Vec::new();

    // Spawn agents across a "world" (200x200 units)
    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 200.0;
        let y = (rng_f64(&mut rng) - 0.5) * 200.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.05;
        }
        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );

        // Diverse harmony profiles — 6 ideological groups
        if let Some(e) = consciousness.entities.get_mut(&h) {
            match i % 6 {
                0 => e.harmony_activations = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5], // traditionalist
                1 => e.harmony_activations = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.5], // progressive
                2 => e.harmony_activations = [0.1, 0.1, 0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.5], // intellectual
                3 => e.harmony_activations = [0.1, 0.1, 0.1, 0.9, 0.8, 0.1, 0.1, 0.1, 0.5], // creative
                4 => e.harmony_activations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], // centrist
                _ => e.harmony_activations = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5], // apathetic
            }
        }
        handles.push(h);
    }

    // Invisible influencers: emit harmony fields but aren't real agents.
    // Models: algorithms, bots, media narratives that shape the field.
    let mut influencer_fields: Vec<([f64; 9], SVector<f64, 2>)> = Vec::new();
    for i in 0..era.influencer_count {
        let x = (rng_f64(&mut rng) - 0.5) * 150.0;
        let y = (rng_f64(&mut rng) - 0.5) * 150.0;
        // Influencers have EXTREME harmony profiles (polarizing)
        let mut harm = [0.0; 9];
        harm[i % 8] = 1.0; // single-issue amplifier
        influencer_fields.push((harm, SVector::from([x, y])));
    }

    let mut cooperation_events = 0u64;
    let mut total_collapses = 0u64;

    for _tick in 0..TICKS {
        // Consciousness update
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

        // FEP gradient
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
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < era.harmony_range // only sense within range
                })
                .cloned()
                .collect();
            let dir = fep_gradient::free_energy_gradient(
                &pos,
                e.energy.fraction_remaining(),
                &e.harmony_activations,
                &near,
                &[],
                None,
                0.0,
            );
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = dir * 20.0;
            }
        }

        // Maintenance + ambient
        let rm = consciousness.resource_regeneration_multiplier();
        for &h in &handles {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
                let phi = e.phi();
                e.energy.consume(era.maintenance * (1.0 + phi * 0.5));
                e.energy
                    .regenerate(consciousness.constants.ambient_regen_rate * rm);
            }
        }

        // Cooperation with resonance threshold (echo chamber effect)
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);

                // Check distance within harmony range
                let in_range = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => {
                        a.position().metric_distance(&b.position()) < era.harmony_range
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

                // Echo chamber: only cooperate with highly resonant agents
                if res > era.resonance_threshold {
                    let rg = consciousness.constants.harmony_resonance_regen_rate
                        * (res - era.resonance_threshold)
                        * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.regenerate(rg);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.regenerate(rg);
                    }
                    cooperation_events += 1;
                }
            }
        }

        // Influencer field effect: shift agent harmonies toward influencer profiles
        for &h in &handles {
            if let Some(body) = world.body(h) {
                let pos = body.position();
                for (inf_harm, inf_pos) in &influencer_fields {
                    let dist = (pos - inf_pos).norm();
                    if dist < era.harmony_range && dist > 1.0 {
                        let influence = era.influencer_strength / (dist * dist + 1.0);
                        if let Some(e) = consciousness.entities.get_mut(&h) {
                            for k in 0..8 {
                                // Gradually shift toward influencer's harmony
                                e.harmony_activations[k] +=
                                    (inf_harm[k] - e.harmony_activations[k]) * influence * 0.01;
                                e.harmony_activations[k] = e.harmony_activations[k].clamp(0.0, 1.0);
                            }
                        }
                    }
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();

        // Track collapses
        for &h in &handles {
            if consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(false)
            {
                total_collapses += 1;
            }
        }
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
        .count();
    let ae = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;

    // Clustering
    let clustering = avg_nn(&world, &handles);

    // Count distinct clusters (connected components with resonance > threshold)
    let num_clusters = count_clusters(
        &world,
        &consciousness,
        &handles,
        era.resonance_threshold,
        era.harmony_range,
    );

    // Polarization: variance of harmony activations across agents
    let polarization = compute_polarization(&consciousness, &handles);

    // Burnout rate: fraction of total agent-ticks that were collapsed
    let burnout_rate = total_collapses as f64 / (AGENTS * TICKS) as f64;

    EraResult {
        alive: alive as f64,
        energy: ae,
        clustering,
        num_clusters,
        polarization,
        cooperation_events: cooperation_events as f64,
        burnout_rate,
    }
}

fn main() {
    eprintln!("=== Internet Effect Experiment ===");
    eprintln!("What has technology done to consciousness-cooperation?");
    eprintln!("Agents: {AGENTS}, Ticks: {TICKS}, Seeds: {SEEDS}\n");

    let eras = vec![
        Era {
            name: "PRE-INTERNET",
            harmony_range: 40.0,
            maintenance: 0.15,
            resonance_threshold: 0.3,
            influencer_count: 0,
            influencer_strength: 0.0,
        },
        Era {
            name: "EARLY-WEB",
            harmony_range: 100.0,
            maintenance: 0.20,
            resonance_threshold: 0.4,
            influencer_count: 2,
            influencer_strength: 5.0,
        },
        Era {
            name: "SOCIAL-MEDIA",
            harmony_range: 300.0,
            maintenance: 0.30,
            resonance_threshold: 0.6,
            influencer_count: 6,
            influencer_strength: 15.0,
        },
        Era {
            name: "ALGO-FEEDS",
            harmony_range: 400.0,
            maintenance: 0.40,
            resonance_threshold: 0.8,
            influencer_count: 10,
            influencer_strength: 25.0,
        },
        Era {
            name: "POST-ALGO",
            harmony_range: 200.0,
            maintenance: 0.25,
            resonance_threshold: 0.4,
            influencer_count: 3,
            influencer_strength: 5.0,
        },
    ];

    println!("era,seed,alive,energy,clustering,clusters,polarization,cooperation,burnout");

    let mut era_results: Vec<(&str, Vec<EraResult>)> = Vec::new();

    for era in &eras {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {} seed={seed}...", era.name);
            let r = run_era(era, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.2},{:.1},{:.4},{:.0},{:.4}",
                era.name,
                r.alive,
                r.energy,
                r.clustering,
                r.num_clusters,
                r.polarization,
                r.cooperation_events,
                r.burnout_rate
            );
            results.push(r);
        }

        let n = results.len() as f64;
        eprintln!("\n── {} ──", era.name);
        eprintln!(
            "  Alive:        {:.1}/{AGENTS}",
            results.iter().map(|r| r.alive).sum::<f64>() / n
        );
        eprintln!(
            "  Energy:       {:.1} J",
            results.iter().map(|r| r.energy).sum::<f64>() / n
        );
        eprintln!(
            "  Clustering:   {:.2}",
            results.iter().map(|r| r.clustering).sum::<f64>() / n
        );
        eprintln!(
            "  Clusters:     {:.1}",
            results.iter().map(|r| r.num_clusters).sum::<f64>() / n
        );
        eprintln!(
            "  Polarization: {:.4}",
            results.iter().map(|r| r.polarization).sum::<f64>() / n
        );
        eprintln!(
            "  Cooperation:  {:.0}",
            results.iter().map(|r| r.cooperation_events).sum::<f64>() / n
        );
        eprintln!(
            "  Burnout rate: {:.2}%",
            results.iter().map(|r| r.burnout_rate).sum::<f64>() / n * 100.0
        );

        era_results.push((era.name, results));
    }

    // Statistical comparison: PRE-INTERNET vs ALGO-FEEDS
    if era_results.len() >= 4 {
        let pre = &era_results[0].1;
        let algo = &era_results[3].1;

        let pre_cluster: Vec<f64> = pre.iter().map(|r| r.clustering).collect();
        let algo_cluster: Vec<f64> = algo.iter().map(|r| r.clustering).collect();
        let (u, z, p) = mann_whitney_u(&pre_cluster, &algo_cluster);
        let d = cohens_d(&pre_cluster, &algo_cluster);

        eprintln!("\n── PRE-INTERNET vs ALGO-FEEDS ──");
        eprintln!(
            "  Clustering: Mann-Whitney U={:.1}, z={:.3}, p={:.4}",
            u, z, p
        );
        eprintln!(
            "  Cohen's d={:.3} ({})",
            d,
            if d.abs() > 0.8 {
                "large"
            } else if d.abs() > 0.5 {
                "medium"
            } else {
                "small"
            }
        );
        eprintln!(
            "  {}",
            if p < 0.05 {
                "SIGNIFICANT"
            } else {
                "not significant"
            }
        );

        let pre_polar: Vec<f64> = pre.iter().map(|r| r.polarization).collect();
        let algo_polar: Vec<f64> = algo.iter().map(|r| r.polarization).collect();
        let (u2, z2, p2) = mann_whitney_u(&pre_polar, &algo_polar);
        let d2 = cohens_d(&pre_polar, &algo_polar);

        eprintln!(
            "  Polarization: Mann-Whitney U={:.1}, z={:.3}, p={:.4}",
            u2, z2, p2
        );
        eprintln!(
            "  Cohen's d={:.3} ({})",
            d2,
            if d2.abs() > 0.8 {
                "large"
            } else {
                "medium/small"
            }
        );

        let pre_burnout: Vec<f64> = pre.iter().map(|r| r.burnout_rate).collect();
        let algo_burnout: Vec<f64> = algo.iter().map(|r| r.burnout_rate).collect();
        let (u3, z3, p3) = mann_whitney_u(&pre_burnout, &algo_burnout);
        let d3 = cohens_d(&pre_burnout, &algo_burnout);

        eprintln!(
            "  Burnout: Mann-Whitney U={:.1}, z={:.3}, p={:.4}",
            u3, z3, p3
        );
        eprintln!(
            "  Cohen's d={:.3} ({})",
            d3,
            if d3.abs() > 0.8 {
                "large"
            } else {
                "medium/small"
            }
        );
    }

    eprintln!("\n=== Complete ===");
}

fn avg_nn(world: &PhysicsWorld<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position()))
        .collect();
    if pos.len() < 2 {
        return f64::MAX;
    }
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
}

fn count_clusters(
    world: &PhysicsWorld<2>,
    consciousness: &ConsciousnessField<2>,
    handles: &[symtropy_physics::BodyHandle],
    threshold: f64,
    range: f64,
) -> f64 {
    let n = handles.len();
    let mut visited = vec![false; n];
    let mut clusters = 0.0;

    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        clusters += 1.0;
        let mut stack = vec![start];
        while let Some(idx) = stack.pop() {
            for other in 0..n {
                if visited[other] {
                    continue;
                }
                let in_range = match (world.body(handles[idx]), world.body(handles[other])) {
                    (Some(a), Some(b)) => a.position().metric_distance(&b.position()) < range,
                    _ => false,
                };
                if !in_range {
                    continue;
                }
                let res = match (
                    consciousness.entities.get(&handles[idx]),
                    consciousness.entities.get(&handles[other]),
                ) {
                    (Some(a), Some(b)) => {
                        HarmonyField::<2>::resonance(&a.harmony_activations, &b.harmony_activations)
                    }
                    _ => 0.0,
                };
                if res > threshold {
                    visited[other] = true;
                    stack.push(other);
                }
            }
        }
    }
    clusters
}

fn compute_polarization(
    consciousness: &ConsciousnessField<2>,
    handles: &[symtropy_physics::BodyHandle],
) -> f64 {
    // Mean harmony vector
    let n = handles.len() as f64;
    let mut mean = [0.0f64; 9];
    for &h in handles {
        if let Some(e) = consciousness.entities.get(&h) {
            for i in 0..8 {
                mean[i] += e.harmony_activations[i];
            }
        }
    }
    for v in &mut mean {
        *v /= n;
    }

    // Variance = mean squared distance from mean
    let mut var = 0.0;
    for &h in handles {
        if let Some(e) = consciousness.entities.get(&h) {
            let dist_sq: f64 = (0..8)
                .map(|i| (e.harmony_activations[i] - mean[i]).powi(2))
                .sum();
            var += dist_sq;
        }
    }
    var / n
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
