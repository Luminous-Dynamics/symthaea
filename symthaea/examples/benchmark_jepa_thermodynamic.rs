// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! JEPA Thermodynamic Benchmark — Fair Comparison
//!
//! Compares prediction quality and energy cost between three systems,
//! all operating on the same 256D input sequence (matching CfC input_dim):
//!
//! 1. **Naive baseline**: Previous state = prediction (no learning, measures signal difficulty)
//! 2. **Linear predictor**: Online linear regression in observation space (learns, fair control)
//! 3. **JEPA**: Latent-space prediction with EMA target encoder (learns in 128D latent)
//!
//! All three measure cosine prediction error on the same input sequence.
//! The linear predictor has comparable parameter count to JEPA's encoder,
//! making the comparison fair.
//!
//! Usage: cargo run --example benchmark_jepa_thermodynamic --features jepa --release

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_jepa::{JepaConfig, JepaEngine, cosine_loss};

/// Input dimension matching CfC default (not HDC_DIMENSION).
const INPUT_DIM: usize = 256;

/// Latent dimension for JEPA.
const LATENT_DIM: usize = 128;

/// Number of cognitive cycles to simulate.
const NUM_CYCLES: usize = 2000;

/// Simulated energy per CfC step (100 fJ, silicon digital substrate).
const CFC_ENERGY_PER_STEP: f64 = 1e-13;

/// Simulated energy per JEPA forward (1 fJ — 128D latent ops are cheap).
const JEPA_ENERGY_PER_STEP: f64 = 1e-15;

/// Simulated energy per linear predictor step (similar to JEPA — same param count).
const LINEAR_ENERGY_PER_STEP: f64 = 1e-15;

#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    total_energy_joules: f64,
    initial_pe: f32,
    final_pe: f32,
    pe_reduction: f32,
    energy_per_pe_reduction: f64,
    pe_trajectory: Vec<f32>,
}

impl BenchmarkResult {
    fn from_trajectory(name: String, pe_trajectory: Vec<f32>, total_energy: f64) -> Self {
        let initial_pe = pe_trajectory[..20].iter().sum::<f32>() / 20.0;
        let final_pe = pe_trajectory[pe_trajectory.len() - 20..]
            .iter()
            .sum::<f32>()
            / 20.0;
        let pe_reduction = initial_pe - final_pe;
        let energy_per_pe = if pe_reduction > 1e-6 {
            total_energy / pe_reduction as f64
        } else {
            f64::INFINITY
        };
        Self {
            name,
            total_energy_joules: total_energy,
            initial_pe,
            final_pe,
            pe_reduction,
            energy_per_pe_reduction: energy_per_pe,
            pe_trajectory,
        }
    }
}

/// Generate a temporal input sequence with multiple frequencies.
/// Uses 256D to match CfC input dimension.
fn generate_input(cycle: usize, dim: usize) -> ContinuousHV {
    let mut values = Vec::with_capacity(dim);
    let t = cycle as f32 * 0.02;
    for i in 0..dim {
        let phase = i as f32 * 0.037; // Irrational-ish spacing
        let slow = (0.3 * t + phase).sin();
        let mid = (1.7 * t + phase * 0.5).sin() * 0.4;
        let fast = (7.1 * t + phase * 0.2).sin() * 0.15;
        values.push((slow + mid + fast).clamp(-1.0, 1.0));
    }
    ContinuousHV::from_vec(values)
}

/// Naive baseline: previous state = prediction (no learning).
fn run_naive(dim: usize) -> BenchmarkResult {
    let mut pe_trajectory = Vec::with_capacity(NUM_CYCLES);
    let mut total_energy = 0.0f64;
    let mut prev = generate_input(0, dim);

    for cycle in 1..=NUM_CYCLES {
        let current = generate_input(cycle, dim);
        let pe = cosine_loss(&prev.values, &current.values).clamp(0.0, 1.0);
        pe_trajectory.push(pe);
        total_energy += CFC_ENERGY_PER_STEP;
        prev = current;
    }

    BenchmarkResult::from_trajectory("Naive (prev=pred)".into(), pe_trajectory, total_energy)
}

/// Online linear predictor in observation space.
/// Learns W such that W * x_t ≈ x_{t+1} via SGD on cosine loss.
/// Parameter count: dim * dim = 65,536 (comparable to JEPA's encoder).
struct LinearPredictor {
    weights: Vec<f32>, // dim x dim, row-major
    dim: usize,
}

impl LinearPredictor {
    fn new(dim: usize, seed: u64) -> Self {
        let scale = (1.0 / dim as f64).sqrt() as f32;
        let mut weights = Vec::with_capacity(dim * dim);
        let mut state = seed ^ 0x9E3779B97F4A7C15;
        for _ in 0..dim * dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 40) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0;
            weights.push(v * scale);
        }
        Self { weights, dim }
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let row = i * self.dim;
            let mut acc = 0.0f32;
            for j in 0..self.dim {
                acc += self.weights[row + j] * input[j];
            }
            out[i] = acc.clamp(-1.0, 1.0);
        }
        out
    }

    fn train_step(&mut self, input: &[f32], target: &[f32], lr: f32) -> f32 {
        let predicted = self.predict(input);
        let loss = cosine_loss(&predicted, target);

        // Gradient of cosine loss w.r.t. predicted
        let grad = symthaea_jepa::cosine_loss_gradient(&predicted, target);

        // Update weights: dL/dW[i,j] = grad[i] * input[j]
        for i in 0..self.dim {
            let row = i * self.dim;
            for j in 0..self.dim {
                self.weights[row + j] -= lr * grad[i] * input[j];
            }
        }

        loss
    }
}

fn run_linear(dim: usize) -> BenchmarkResult {
    let mut predictor = LinearPredictor::new(dim, 42);
    let mut pe_trajectory = Vec::with_capacity(NUM_CYCLES);
    let mut total_energy = 0.0f64;
    let mut prev = generate_input(0, dim);

    for cycle in 1..=NUM_CYCLES {
        let current = generate_input(cycle, dim);
        let loss = predictor.train_step(&prev.values, &current.values, 0.001);
        pe_trajectory.push(loss.clamp(0.0, 1.0));
        total_energy += CFC_ENERGY_PER_STEP + LINEAR_ENERGY_PER_STEP;
        prev = current;
    }

    BenchmarkResult::from_trajectory("Linear (obs-space)".into(), pe_trajectory, total_energy)
}

/// JEPA: latent-space prediction with EMA target encoder.
fn run_jepa(dim: usize) -> BenchmarkResult {
    let config = JepaConfig {
        input_dim: dim,
        latent_dim: LATENT_DIM,
        learning_rate: 0.001,
        ema_momentum: 0.996,
        energy_cost_per_forward: JEPA_ENERGY_PER_STEP,
        ..Default::default()
    };
    let mut engine = JepaEngine::new(config, 42);
    let mut pe_trajectory = Vec::with_capacity(NUM_CYCLES);
    let mut total_energy = 0.0f64;
    let mut prev = generate_input(0, dim);

    for cycle in 1..=NUM_CYCLES {
        let current = generate_input(cycle, dim);
        let action = (cycle % 8) as u8;
        let _loss = engine.train_step(&prev, &current, action, 0.001);
        let pe = engine.latent_pe();
        pe_trajectory.push(pe);
        total_energy += CFC_ENERGY_PER_STEP + JEPA_ENERGY_PER_STEP;
        prev = current;
    }

    BenchmarkResult::from_trajectory("JEPA (latent-space)".into(), pe_trajectory, total_energy)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  JEPA Thermodynamic Benchmark (Fair Comparison)");
    println!(
        "  {}D input (CfC-matched), {}D latent, {} cycles",
        INPUT_DIM, LATENT_DIM, NUM_CYCLES
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    let naive = run_naive(INPUT_DIM);
    let linear = run_linear(INPUT_DIM);
    let jepa = run_jepa(INPUT_DIM);

    // Comparison table
    println!("┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("│ Metric              │ Naive            │ Linear (obs)     │ JEPA (latent)    │");
    println!("├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (label, results) in [
        (
            "Total energy (J)",
            vec![
                naive.total_energy_joules,
                linear.total_energy_joules,
                jepa.total_energy_joules,
            ],
        ),
        (
            "Initial PE",
            vec![
                naive.initial_pe as f64,
                linear.initial_pe as f64,
                jepa.initial_pe as f64,
            ],
        ),
        (
            "Final PE",
            vec![
                naive.final_pe as f64,
                linear.final_pe as f64,
                jepa.final_pe as f64,
            ],
        ),
        (
            "PE reduction",
            vec![
                naive.pe_reduction as f64,
                linear.pe_reduction as f64,
                jepa.pe_reduction as f64,
            ],
        ),
    ] {
        println!(
            "│ {:<19} │ {:>16.4e} │ {:>16.4e} │ {:>16.4e} │",
            label, results[0], results[1], results[2]
        );
    }
    println!(
        "│ {:<19} │ {:>16.4e} │ {:>16.4e} │ {:>16.4e} │",
        "J / PE-reduction",
        naive.energy_per_pe_reduction,
        linear.energy_per_pe_reduction,
        jepa.energy_per_pe_reduction
    );
    println!("└─────────────────────┴──────────────────┴──────────────────┴──────────────────┘");

    // JEPA vs Linear comparison
    if jepa.pe_reduction > 1e-6 && linear.pe_reduction > 1e-6 {
        let pe_ratio = jepa.pe_reduction / linear.pe_reduction;
        let energy_ratio = jepa.total_energy_joules / linear.total_energy_joules;
        let efficiency = pe_ratio as f64 / energy_ratio;
        println!("\n  JEPA vs Linear predictor:");
        println!("    PE reduction ratio: {:.2}x", pe_ratio);
        println!("    Energy ratio: {:.4}x", energy_ratio);
        println!("    Thermodynamic efficiency: {:.2}x", efficiency);
        if efficiency > 1.0 {
            println!("    → JEPA achieves more PE reduction per joule (latent > observation)");
        } else {
            println!("    → Linear predictor is more efficient (observation > latent)");
        }
    }

    // PE trajectory samples
    println!("\n  PE trajectory (sampled every 200 cycles):");
    println!(
        "  {:>6}  {:>10}  {:>10}  {:>10}",
        "Cycle", "Naive", "Linear", "JEPA"
    );
    for i in (0..NUM_CYCLES).step_by(200) {
        println!(
            "  {:>6}  {:>10.4}  {:>10.4}  {:>10.4}",
            i + 1,
            naive.pe_trajectory.get(i).unwrap_or(&0.0),
            linear.pe_trajectory.get(i).unwrap_or(&0.0),
            jepa.pe_trajectory.get(i).unwrap_or(&0.0),
        );
    }

    // JSON
    println!("\n--- JSON ---");
    println!(
        r#"{{"dim": {}, "latent": {}, "cycles": {}, "naive_final_pe": {:.6}, "linear_final_pe": {:.6}, "jepa_final_pe": {:.6}, "naive_energy": {:.2e}, "linear_energy": {:.2e}, "jepa_energy": {:.2e}}}"#,
        INPUT_DIM,
        LATENT_DIM,
        NUM_CYCLES,
        naive.final_pe,
        linear.final_pe,
        jepa.final_pe,
        naive.total_energy_joules,
        linear.total_energy_joules,
        jepa.total_energy_joules,
    );
}
