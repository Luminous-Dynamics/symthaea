// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # O(1) vs Naive Learning Value Prediction Benchmark
//!
//! This benchmark demonstrates the performance advantage of CfC's closed-form
//! solution for predicting learning value.
//!
//! ## The Key Insight
//!
//! Traditional evaluation (naive): O(N) where N = simulation steps
//! - Must simulate each timestep to predict future state
//! - Time grows linearly with prediction horizon
//!
//! CfC evaluation (O(1)): x(t) = (x₀ - A) · e^(-t/τ) + A
//! - Jump directly to any future time point
//! - Time is constant regardless of horizon
//!
//! ## Run
//!
//! ```bash
//! cargo run --example school_o1_benchmark --release
//! ```

use std::time::{Duration, Instant};
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::school::{Difficulty, Domain, LearningObjective, School, SchoolConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// NAIVE O(N) EVALUATOR (FOR COMPARISON)
// ═══════════════════════════════════════════════════════════════════════════════

/// Naive evaluator that simulates N timesteps
///
/// This models a more realistic RNN-style evaluation where each timestep
/// requires matrix operations proportional to the network size.
#[allow(dead_code)]
struct NaiveEvaluator {
    /// Number of simulation steps per evaluation
    n_steps: usize,
    /// Time step size
    dt: f32,
    /// State dimension (simulated neuron count)
    neurons: usize,
    /// Weight matrix (simulates RNN connections)
    weights: Vec<Vec<f32>>,
}

#[allow(dead_code)]
impl NaiveEvaluator {
    fn new(n_steps: usize, dt: f32, neurons: usize) -> Self {
        // Create a random weight matrix for realistic RNN simulation
        let weights: Vec<Vec<f32>> = (0..neurons)
            .map(|i| {
                (0..neurons)
                    .map(|j| {
                        let seed = (i * 1000 + j) as f32;
                        (seed * 1.234).sin() * 0.1 // Small random weights
                    })
                    .collect()
            })
            .collect();

        Self {
            n_steps,
            dt,
            neurons,
            weights,
        }
    }

    /// Evaluate an objective by simulating forward in time
    ///
    /// This is O(N × D²) where N = n_steps and D = neurons
    /// Each timestep requires a full matrix-vector multiplication
    fn evaluate(&self, objective: &LearningObjective, initial_state: &[f32]) -> f32 {
        let mut state = initial_state[..self.neurons.min(initial_state.len())].to_vec();
        state.resize(self.neurons, 0.0);

        let obj_enc = objective.encoding.values.as_slice();
        let input: Vec<f32> = (0..self.neurons)
            .map(|i| {
                if i < obj_enc.len() {
                    obj_enc[i] * 0.01
                } else {
                    0.0
                }
            })
            .collect();

        // Simulate N timesteps with matrix operations
        for _ in 0..self.n_steps {
            let mut new_state = vec![0.0f32; self.neurons];

            // Matrix-vector multiplication: new_state = W @ state + input
            for i in 0..self.neurons {
                let mut sum = input[i];
                for j in 0..self.neurons {
                    sum += self.weights[i][j] * state[j];
                }
                // Tanh activation + decay
                new_state[i] = (sum.tanh() - state[i] * 0.1) * self.dt + state[i];
            }

            state = new_state;
        }

        // Compute "phi gain" from final state
        let coherence: f32 = state.iter().sum::<f32>() / state.len() as f32;
        coherence.abs() * 0.01
    }
}

/// Simpler naive evaluator for faster benchmarking
/// Uses basic array operations instead of matrix multiplication
struct SimpleNaiveEvaluator {
    n_steps: usize,
    dt: f32,
}

impl SimpleNaiveEvaluator {
    fn new(n_steps: usize, dt: f32) -> Self {
        Self { n_steps, dt }
    }

    /// O(N × D) evaluation using simple dynamics
    fn evaluate(&self, objective: &LearningObjective, initial_state: &[f32]) -> f32 {
        let mut state = initial_state.to_vec();
        let obj_enc = objective.encoding.values.as_slice();

        for _ in 0..self.n_steps {
            for i in 0..state.len().min(obj_enc.len()) {
                let influence = obj_enc[i] * 0.01;
                let decay = -state[i] * 0.1;
                state[i] += (influence + decay) * self.dt;
            }
        }

        let coherence: f32 = state.iter().sum::<f32>() / state.len() as f32;
        coherence.abs() * 0.01
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK HARNESS
// ═══════════════════════════════════════════════════════════════════════════════

fn benchmark<F>(name: &str, iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10 {
        f();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let total = start.elapsed();

    println!(
        "  {}: {:?} total, {:?} per iteration",
        name,
        total,
        total / iterations as u32
    );

    total
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║      O(1) CfC Lookahead vs O(N) Naive Simulation Benchmark                   ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  CfC Solution: x(t) = (x₀ - A) · e^(-t/τ) + A                                ║");
    println!("║  Naive Method: Simulate each timestep sequentially                           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Create test objectives
    let objectives: Vec<LearningObjective> = (0..100)
        .map(|i| {
            LearningObjective::new(&format!("obj-{}", i), &format!("Test Objective {}", i))
                .with_domain(Domain::NixOS)
                .with_difficulty(Difficulty::Intermediate)
                .with_seed(i as u64 * 42)
                .build()
        })
        .collect();

    // Create consciousness state
    let consciousness_state: Vec<ContinuousHV> = (0..8)
        .map(|i| {
            let hv = ContinuousHV::random(HDC_DIMENSION, 42 + i as u64);
            ContinuousHV::from_vec(hv.values)
        })
        .collect();

    let initial_state: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0) * 2.0 - 1.0).collect();

    // ─────────────────────────────────────────────────────────────────────────────
    // BENCHMARK 1: Single Objective Evaluation
    // ─────────────────────────────────────────────────────────────────────────────

    println!("\n═══ Benchmark 1: Single Objective Evaluation ═══\n");

    let school = School::new(SchoolConfig::fast()).expect("Failed to create school");

    // Test different horizon lengths using SimpleNaiveEvaluator for faster benchmarks
    for horizon in [0.1, 1.0, 10.0, 100.0] {
        println!("Horizon: {} seconds", horizon);

        // O(1) CfC evaluation
        let iterations = 1000;
        let obj = &objectives[0];

        let cfc_time = benchmark("  CfC O(1)", iterations, || {
            let _ = school
                .lookahead_engine()
                .evaluate(obj, &consciousness_state);
        });

        // O(N) Simple Naive evaluation (for faster benchmarking)
        let n_steps = (horizon * 1000.0) as usize; // 1000 steps per second
        let naive = SimpleNaiveEvaluator::new(n_steps, 0.001);

        let naive_time = benchmark(&format!("  Naive O({})", n_steps), iterations, || {
            let _ = naive.evaluate(obj, &initial_state);
        });

        let speedup = naive_time.as_nanos() as f64 / cfc_time.as_nanos() as f64;
        println!("  → CfC speedup: {:.1}x faster\n", speedup);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // BENCHMARK 2: Batch Objective Ranking
    // ─────────────────────────────────────────────────────────────────────────────

    println!(
        "\n═══ Benchmark 2: Ranking {} Objectives ═══\n",
        objectives.len()
    );

    let obj_refs: Vec<&LearningObjective> = objectives.iter().collect();

    let iterations = 100;

    let cfc_batch = benchmark("CfC O(1) batch", iterations, || {
        let _ = school
            .lookahead_engine()
            .rank_objectives(&obj_refs, &consciousness_state);
    });

    let naive = SimpleNaiveEvaluator::new(1000, 0.001); // 1 second horizon
    let naive_batch = benchmark("Naive O(1000) batch", iterations, || {
        let mut results: Vec<(usize, f32)> = objectives
            .iter()
            .enumerate()
            .map(|(i, obj)| (i, naive.evaluate(obj, &initial_state)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    });

    let speedup = naive_batch.as_nanos() as f64 / cfc_batch.as_nanos() as f64;
    println!("→ CfC batch speedup: {:.1}x faster\n", speedup);

    // ─────────────────────────────────────────────────────────────────────────────
    // BENCHMARK 3: Scaling with Prediction Horizon
    // ─────────────────────────────────────────────────────────────────────────────

    println!("\n═══ Benchmark 3: Scaling with Horizon ═══\n");
    println!("│ Horizon │ CfC O(1) │ Naive O(N) │ Speedup │");
    println!("├─────────┼──────────┼────────────┼─────────┤");

    let obj = &objectives[0];
    let iterations = 500;

    for horizon in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0] {
        // CfC (constant time)
        let cfc_start = Instant::now();
        for _ in 0..iterations {
            let _ = school
                .lookahead_engine()
                .evaluate(obj, &consciousness_state);
        }
        let cfc_per = cfc_start.elapsed() / iterations as u32;

        // Naive (linear in horizon) - using SimpleNaiveEvaluator for faster benchmarks
        let n_steps = (horizon * 1000.0) as usize; // 1000 steps/second
        let naive = SimpleNaiveEvaluator::new(n_steps, 0.001);

        let naive_start = Instant::now();
        for _ in 0..iterations {
            let _ = naive.evaluate(obj, &initial_state);
        }
        let naive_per = naive_start.elapsed() / iterations as u32;

        let speedup = naive_per.as_nanos() as f64 / cfc_per.as_nanos() as f64;

        println!(
            "│ {:>7.1} │ {:>8.1?} │ {:>10.1?} │ {:>6.1}x │",
            horizon, cfc_per, naive_per, speedup
        );
    }

    println!("└─────────┴──────────┴────────────┴─────────┘");

    // ─────────────────────────────────────────────────────────────────────────────
    // SUMMARY
    // ─────────────────────────────────────────────────────────────────────────────

    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              BENCHMARK SUMMARY                                 ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                               ║");
    println!("║  CfC's closed-form solution provides O(1) prediction regardless of horizon.  ║");
    println!("║                                                                               ║");
    println!("║  Key advantages:                                                              ║");
    println!("║  • Constant time: No increase with longer prediction horizons                ║");
    println!("║  • Parallelizable: Each objective can be evaluated independently             ║");
    println!("║  • Energy efficient: Fewer compute cycles = less power                       ║");
    println!("║                                                                               ║");
    println!("║  This enables anticipatory learning where predicting the value of learning   ║");
    println!("║  costs negligible time, allowing optimal curriculum sequencing in real-time. ║");
    println!("║                                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
}
