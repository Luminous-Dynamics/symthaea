// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Swarm Consciousness Emergence Experiment
//!
//! Tests whether collective consciousness emerges from groups of interacting agents.
//! Measures collective Phi (via pairwise thought coherence), emergence ratios,
//! and Byzantine resilience.
//!
//! ## Metrics
//!
//! - **Individual Phi**: Each agent's consciousness_level
//! - **Collective coherence**: Mean pairwise thought similarity
//! - **Emergence ratio**: collective_metric / mean(individual_metric)
//!   - ratio > 1.0 implies emergent collective consciousness
//!
//! ## Usage
//!
//! ```bash
//! # Single run with 10 agents
//! cargo run --example swarm_consciousness_emergence -- --agents 10 --ticks 200
//!
//! # Scaling experiment
//! cargo run --example swarm_consciousness_emergence -- --scaling
//!
//! # Byzantine resilience (30% poisoned agents)
//! cargo run --example swarm_consciousness_emergence -- --agents 10 --byzantine 0.3
//! ```

use std::time::Duration;

use symthaea::mind::{AsyncMind, MindConfig, MindState, connect_social};
use symthaea_core::hdc::ContinuousHV;

// ============================================================================
// METRICS
// ============================================================================

struct SwarmSnapshot {
    tick: u64,
    n_agents: usize,
    n_byzantine: usize,
    individual_consciousness: Vec<f64>,
    pairwise_similarities: Vec<f32>,
}

impl SwarmSnapshot {
    fn mean_consciousness(&self) -> f64 {
        if self.individual_consciousness.is_empty() {
            return 0.0;
        }
        self.individual_consciousness.iter().sum::<f64>()
            / self.individual_consciousness.len() as f64
    }

    fn mean_coherence(&self) -> f32 {
        if self.pairwise_similarities.is_empty() {
            return 0.0;
        }
        self.pairwise_similarities.iter().sum::<f32>() / self.pairwise_similarities.len() as f32
    }

    fn max_coherence(&self) -> f32 {
        self.pairwise_similarities
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max)
    }

    fn min_coherence(&self) -> f32 {
        self.pairwise_similarities
            .iter()
            .cloned()
            .fold(f32::MAX, f32::min)
    }

    fn emergence_ratio(&self) -> f64 {
        let mean_c = self.mean_consciousness();
        if mean_c < 0.001 {
            return 0.0;
        }
        // Emergence: collective coherence normalized by individual consciousness
        // Higher coherence relative to individual consciousness = more emergence
        self.mean_coherence() as f64 / mean_c
    }

    fn csv_header() -> &'static str {
        "tick,n_agents,n_byzantine,mean_consciousness,mean_coherence,max_coherence,min_coherence,emergence_ratio"
    }

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            self.tick,
            self.n_agents,
            self.n_byzantine,
            self.mean_consciousness(),
            self.mean_coherence() as f64,
            self.max_coherence() as f64,
            self.min_coherence() as f64,
            self.emergence_ratio(),
        )
    }
}

fn compute_snapshot(tick: u64, states: &[MindState], n_byzantine: usize) -> SwarmSnapshot {
    let n = states.len();
    let individual_consciousness: Vec<f64> = states.iter().map(|s| s.consciousness_level).collect();

    let mut pairwise = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            // Guard against dimension mismatch (cognitive loop may expand dims)
            if states[i].current_thought.dim() == states[j].current_thought.dim() {
                pairwise.push(
                    states[i]
                        .current_thought
                        .similarity(&states[j].current_thought),
                );
            } else {
                // Truncate to smaller dimension for comparison
                let min_dim = states[i]
                    .current_thought
                    .dim()
                    .min(states[j].current_thought.dim());
                let a = ContinuousHV::from_slice(&states[i].current_thought.as_slice()[..min_dim]);
                let b = ContinuousHV::from_slice(&states[j].current_thought.as_slice()[..min_dim]);
                pairwise.push(a.similarity(&b));
            }
        }
    }

    SwarmSnapshot {
        tick,
        n_agents: n,
        n_byzantine,
        individual_consciousness,
        pairwise_similarities: pairwise,
    }
}

// ============================================================================
// EXPERIMENT
// ============================================================================

async fn run_experiment(n_agents: usize, n_ticks: usize, byzantine_fraction: f64, seed: u64) {
    let dim = 512;
    let n_byzantine = (n_agents as f64 * byzantine_fraction) as usize;
    let n_honest = n_agents - n_byzantine;

    eprintln!(
        "Running: {} agents ({} honest, {} byzantine), {} ticks, seed {}",
        n_agents, n_honest, n_byzantine, n_ticks, seed
    );

    let config = MindConfig {
        dimension: dim,
        enable_social_coherence: true,
        social_projection_enabled: true,
        tick_rate: 10.0,
        ..Default::default()
    };

    // Spawn agents
    let mut handles = Vec::with_capacity(n_agents);
    let mut joins = Vec::with_capacity(n_agents);

    for _ in 0..n_agents {
        let (handle, join) = AsyncMind::spawn(config.clone());
        handles.push(handle);
        joins.push(join);
    }

    // Connect social relays
    // For small groups: full mesh. For larger: k-nearest neighbor (k=4)
    let relay_interval = Duration::from_millis(50);
    let mut _relays = Vec::new();

    if n_agents <= 15 {
        // Full mesh
        for i in 0..n_agents {
            for j in (i + 1)..n_agents {
                _relays.push(connect_social(&handles[i], &handles[j], relay_interval));
            }
        }
        eprintln!("Connected: full mesh ({} relays)", _relays.len());
    } else {
        // k-nearest neighbor ring (k=4)
        for i in 0..n_agents {
            for k in 1..=2 {
                let j = (i + k) % n_agents;
                _relays.push(connect_social(&handles[i], &handles[j], relay_interval));
            }
        }
        eprintln!("Connected: 4-nearest ring ({} relays)", _relays.len());
    }

    // CSV header
    println!("{}", SwarmSnapshot::csv_header());

    // Phase boundaries
    let phase1_end = n_ticks / 5; // Independent (20%)
    let phase2_end = n_ticks * 3 / 5; // Shared experience (40%)
    let _phase3_end = n_ticks; // Byzantine resilience (40%)

    for tick in 0..n_ticks as u64 {
        // Determine stimuli
        if tick < phase1_end as u64 {
            // Phase 1: Independent — each agent sees different stimuli
            for (i, handle) in handles.iter().enumerate() {
                handle
                    .perceive(ContinuousHV::random(dim, seed + (i as u64 * 10000) + tick))
                    .await;
            }
        } else if tick < phase2_end as u64 {
            // Phase 2: Shared experience — honest agents see same stimulus
            let shared = ContinuousHV::random(dim, seed + 500000 + tick);
            for (i, handle) in handles.iter().enumerate() {
                if i < n_honest {
                    handle.perceive(shared.clone()).await;
                } else {
                    // Byzantine agents see anti-correlated noise
                    handle
                        .perceive(ContinuousHV::random(dim, seed + 900000 + (i as u64) + tick))
                        .await;
                }
            }
        } else {
            // Phase 3: Byzantine stress — honest agents on shared, byzantine on random
            let shared = ContinuousHV::random(dim, seed + 700000 + tick);
            for (i, handle) in handles.iter().enumerate() {
                if i < n_honest {
                    handle.perceive(shared.clone()).await;
                } else {
                    // Byzantine: rapid random noise
                    handle
                        .perceive(ContinuousHV::random(
                            dim,
                            seed + 800000 + tick * 100 + i as u64,
                        ))
                        .await;
                }
            }
        }

        // Tick all agents
        for handle in &handles {
            handle.tick().await;
        }

        // Brief sleep for relay propagation
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Sample metrics every 5 ticks
        if tick % 5 == 0 || tick == (n_ticks as u64 - 1) {
            let mut states = Vec::with_capacity(n_agents);
            for handle in &handles {
                states.push(handle.snapshot().await);
            }

            let snapshot = compute_snapshot(tick, &states, n_byzantine);
            println!("{}", snapshot.to_csv());

            // Phase transition logging
            if tick == phase1_end as u64 {
                eprintln!(
                    "  Phase 1→2 (tick {}): mean_c={:.3}, coherence={:.3}",
                    tick,
                    snapshot.mean_consciousness(),
                    snapshot.mean_coherence()
                );
            } else if tick == phase2_end as u64 {
                eprintln!(
                    "  Phase 2→3 (tick {}): mean_c={:.3}, coherence={:.3}, emergence={:.3}",
                    tick,
                    snapshot.mean_consciousness(),
                    snapshot.mean_coherence(),
                    snapshot.emergence_ratio()
                );
            }
        }
    }

    // Final summary
    let mut final_states = Vec::with_capacity(n_agents);
    for handle in &handles {
        final_states.push(handle.snapshot().await);
    }
    let final_snapshot = compute_snapshot(n_ticks as u64, &final_states, n_byzantine);

    eprintln!();
    eprintln!("--- Final State ---");
    eprintln!(
        "  Mean consciousness: {:.4}",
        final_snapshot.mean_consciousness()
    );
    eprintln!("  Mean coherence: {:.4}", final_snapshot.mean_coherence());
    eprintln!("  Emergence ratio: {:.4}", final_snapshot.emergence_ratio());

    if n_byzantine > 0 {
        // Separate honest vs byzantine consciousness
        let honest_c: Vec<f64> = final_states[..n_honest]
            .iter()
            .map(|s| s.consciousness_level)
            .collect();
        let byzantine_c: Vec<f64> = final_states[n_honest..]
            .iter()
            .map(|s| s.consciousness_level)
            .collect();
        let mean_honest: f64 = honest_c.iter().sum::<f64>() / honest_c.len().max(1) as f64;
        let mean_byzantine: f64 = if byzantine_c.is_empty() {
            0.0
        } else {
            byzantine_c.iter().sum::<f64>() / byzantine_c.len() as f64
        };
        eprintln!(
            "  Honest consciousness: {:.4}, Byzantine: {:.4}",
            mean_honest, mean_byzantine
        );
    }

    // Shutdown
    drop(_relays);
    for handle in handles {
        handle.shutdown().await;
    }
    for join in joins {
        let _ = join.await;
    }
}

async fn run_scaling(seed: u64) {
    eprintln!("=== Scaling Experiment ===");
    println!("n_agents,mean_consciousness,mean_coherence,emergence_ratio");

    for &n in &[3, 5, 8, 10, 15] {
        let dim = 512;
        let config = MindConfig {
            dimension: dim,
            enable_social_coherence: true,
            tick_rate: 10.0,
            ..Default::default()
        };

        let mut handles = Vec::new();
        let mut joins = Vec::new();
        for _ in 0..n {
            let (h, j) = AsyncMind::spawn(config.clone());
            handles.push(h);
            joins.push(j);
        }

        // Full mesh
        let relay_interval = Duration::from_millis(50);
        let mut _relays = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                _relays.push(connect_social(&handles[i], &handles[j], relay_interval));
            }
        }

        // Shared experience for 100 ticks
        for tick in 0..100u64 {
            let shared = ContinuousHV::random(dim, seed + tick);
            for handle in &handles {
                handle.perceive(shared.clone()).await;
            }
            for handle in &handles {
                handle.tick().await;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let mut states = Vec::new();
        for handle in &handles {
            states.push(handle.snapshot().await);
        }
        let snapshot = compute_snapshot(100, &states, 0);

        println!(
            "{},{:.6},{:.6},{:.6}",
            n,
            snapshot.mean_consciousness(),
            snapshot.mean_coherence(),
            snapshot.emergence_ratio()
        );

        eprintln!(
            "  N={:>3}: consciousness={:.3}, coherence={:.3}, emergence={:.3}",
            n,
            snapshot.mean_consciousness(),
            snapshot.mean_coherence(),
            snapshot.emergence_ratio()
        );

        drop(_relays);
        for handle in handles {
            handle.shutdown().await;
        }
        for join in joins {
            let _ = join.await;
        }
    }
}

// ============================================================================
// PARTIAL-INFORMATION EXPERIMENT
// ============================================================================

/// Run a single partial-information experiment: each agent sees a different fragment
/// of a composite stimulus and must integrate via social relay.
async fn run_partial_experiment(n_agents: usize, n_ticks: usize, seed: u64) -> (f64, f64, u64) {
    let dim = 512;
    let config = MindConfig {
        dimension: dim,
        enable_social_coherence: true,
        social_projection_enabled: true,
        tick_rate: 10.0,
        ..Default::default()
    };

    // Spawn agents
    let mut handles = Vec::new();
    let mut joins = Vec::new();
    for _ in 0..n_agents {
        let (h, j) = AsyncMind::spawn(config.clone());
        handles.push(h);
        joins.push(j);
    }

    // Full mesh for small groups, k-nearest for larger
    let relay_interval = Duration::from_millis(50);
    let mut _relays = Vec::new();
    if n_agents <= 15 {
        for i in 0..n_agents {
            for j in (i + 1)..n_agents {
                _relays.push(connect_social(&handles[i], &handles[j], relay_interval));
            }
        }
    } else {
        for i in 0..n_agents {
            for k in 1..=2 {
                let j = (i + k) % n_agents;
                _relays.push(connect_social(&handles[i], &handles[j], relay_interval));
            }
        }
    }

    // Generate K orthogonal feature channels (one per agent)
    let channels: Vec<ContinuousHV> = (0..n_agents)
        .map(|i| ContinuousHV::random(dim, seed + 100_000 + i as u64))
        .collect();

    let phase1_end = n_ticks / 4; // 25% fragment-only perception
    let phase2_end = n_ticks * 3 / 4; // 50% social integration
    // remaining 25% = convergence measurement

    // CSV header
    println!("tick,n_agents,phase,mean_consciousness,mean_coherence,emergence_ratio");

    let mut convergence_tick = n_ticks as u64;

    for tick in 0..n_ticks as u64 {
        let phase = if tick < phase1_end as u64 {
            "fragment"
        } else if tick < phase2_end as u64 {
            "integration"
        } else {
            "convergence"
        };

        // Each agent perceives its own fragment: weighted_bundle([channel_i × 3, noise × 1])
        for (i, handle) in handles.iter().enumerate() {
            let noise = ContinuousHV::random(dim, seed + tick * 1000 + i as u64);
            let fragment = ContinuousHV::weighted_bundle(
                &[&channels[i], &channels[i], &channels[i], &noise],
                &[0.75, 0.0, 0.0, 0.25], // 75% own channel, 25% noise
            );
            handle.perceive(fragment).await;
        }

        for handle in &handles {
            handle.tick().await;
        }

        tokio::time::sleep(Duration::from_millis(5)).await;

        // Sample every 5 ticks
        if tick % 5 == 0 || tick == (n_ticks as u64 - 1) {
            let mut states = Vec::new();
            for handle in &handles {
                states.push(handle.snapshot().await);
            }
            let snapshot = compute_snapshot(tick, &states, 0);

            println!(
                "{},{},{},{:.6},{:.6},{:.6}",
                tick,
                n_agents,
                phase,
                snapshot.mean_consciousness(),
                snapshot.mean_coherence(),
                snapshot.emergence_ratio()
            );

            // Detect convergence (coherence crosses 0.5 for first time)
            if convergence_tick == n_ticks as u64 && snapshot.mean_coherence() > 0.5 {
                convergence_tick = tick;
            }
        }
    }

    // Final measurement
    let mut final_states = Vec::new();
    for handle in &handles {
        final_states.push(handle.snapshot().await);
    }
    let final_snapshot = compute_snapshot(n_ticks as u64, &final_states, 0);

    let final_coherence = final_snapshot.mean_coherence() as f64;
    let final_emergence = final_snapshot.emergence_ratio();

    eprintln!(
        "  N={}: coherence={:.4}, emergence={:.4}, convergence_tick={}",
        n_agents, final_coherence, final_emergence, convergence_tick
    );

    // Shutdown
    drop(_relays);
    for handle in handles {
        handle.shutdown().await;
    }
    for join in joins {
        let _ = join.await;
    }

    (final_coherence, final_emergence, convergence_tick)
}

async fn run_partial_scaling(seed: u64) {
    eprintln!("=== Partial-Information Scaling Experiment ===");
    println!("n_agents,fragments,final_coherence,final_emergence_ratio,convergence_tick");

    for &n in &[3, 5, 8, 10, 15, 20] {
        eprintln!("Running N={}...", n);
        let (coherence, emergence, conv_tick) = run_partial_experiment(n, 200, seed).await;
        // Print scaling summary line (separate from per-tick output which goes to a different file)
        eprintln!(
            "  N={:>3}: fragments={}, coherence={:.4}, emergence={:.4}, convergence={}",
            n, n, coherence, emergence, conv_tick
        );
    }
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut n_agents = 10;
    let mut n_ticks = 200;
    let mut byzantine = 0.0f64;
    let mut seed = 42u64;
    let mut scaling = false;
    let mut partial = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agents" => {
                i += 1;
                n_agents = args[i].parse().unwrap_or(10);
            }
            "--ticks" => {
                i += 1;
                n_ticks = args[i].parse().unwrap_or(200);
            }
            "--byzantine" => {
                i += 1;
                byzantine = args[i].parse().unwrap_or(0.0);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or(42);
            }
            "--scaling" => {
                scaling = true;
            }
            "--partial" => {
                partial = true;
            }
            "--help" | "-h" => {
                eprintln!("Swarm Consciousness Emergence Experiment");
                eprintln!();
                eprintln!("OPTIONS:");
                eprintln!("  --agents <N>      Number of agents (default: 10)");
                eprintln!("  --ticks <N>       Simulation ticks (default: 200)");
                eprintln!("  --byzantine <F>   Fraction of byzantine agents (default: 0.0)");
                eprintln!("  --seed <N>        Random seed (default: 42)");
                eprintln!("  --scaling         Run scaling experiment (3-15 agents)");
                eprintln!("  --partial         Partial-information mode (semantic fragmentation)");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    if partial && scaling {
        run_partial_scaling(seed).await;
    } else if partial {
        eprintln!("=== Partial-Information Swarm Experiment ===");
        run_partial_experiment(n_agents, n_ticks, seed).await;
    } else if scaling {
        run_scaling(seed).await;
    } else {
        eprintln!("=== Swarm Consciousness Emergence ===");
        run_experiment(n_agents, n_ticks, byzantine, seed).await;
    }
}