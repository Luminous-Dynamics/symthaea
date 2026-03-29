// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Winterfell PoGQ Benchmark Tool
//!
//! Runs N proofs and emits JSON statistics for comparison with RISC Zero.
//!
//! Usage:
//!   winterfell-benchmark --rounds 20 --output results/winterfell_bench.json

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use winterfell_pogq::{AirPublicInputs as PublicInputs, PoGQProver, SCALE, AIR_SCHEMA_REV};

#[derive(Parser)]
#[command(name = "winterfell-benchmark")]
#[command(about = "Benchmark Winterfell PoGQ prover", long_about = None)]
struct Cli {
    /// Number of proof rounds to run
    #[arg(short, long, default_value = "20")]
    rounds: usize,

    /// Output JSON file
    #[arg(short, long, default_value = "results/winterfell_bench.json")]
    output: PathBuf,

    /// Scenario to benchmark (normal, enter_quarantine, release)
    #[arg(short, long, default_value = "normal")]
    scenario: String,

    /// Trace length (number of PoGQ steps, must be power of 2)
    #[arg(short = 't', long, default_value = "8")]
    trace_length: usize,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    backend: String,
    scenario: String,
    trace_length: usize,
    rounds: usize,
    prove_times_ms: Vec<u64>,
    verify_times_ms: Vec<u64>,
    proof_sizes_bytes: Vec<usize>,
    mean_prove_ms: f64,
    std_prove_ms: f64,
    p95_prove_ms: u64,
    mean_verify_ms: f64,
    std_verify_ms: f64,
    p95_verify_ms: u64,
    mean_proof_size_kb: f64,
}

fn q(val: f32) -> u64 {
    (val * SCALE as f32) as u64
}

fn get_scenario_inputs(scenario: &str, trace_length: usize) -> (PublicInputs, Vec<u64>) {
    match scenario {
        "normal" => {
            let public = PublicInputs {
                beta: q(0.85),
                w: 3,
                k: 2,
                m: 3,
                threshold: q(0.90),
                ema_init: q(0.915),
                viol_init: 0,
                clear_init: 2,
                quar_init: 0,
                round_init: 4,
                quar_out: 0, // No quarantine
                trace_length,
                // Provenance (prover will populate)
                prov_hash: [0, 0, 0, 0],
                profile_id: 128,
                air_rev: AIR_SCHEMA_REV,
            };
            let witness = vec![q(0.930); trace_length]; // Above threshold
            (public, witness)
        }
        "enter_quarantine" => {
            let public = PublicInputs {
                beta: q(0.85),
                w: 3,
                k: 2,
                m: 3,
                threshold: q(0.90),
                ema_init: q(0.763),
                viol_init: 1,
                clear_init: 0,
                quar_init: 0,
                round_init: 7,
                quar_out: 1, // Enter quarantine
                trace_length,
                // Provenance (prover will populate)
                prov_hash: [0, 0, 0, 0],
                profile_id: 128,
                air_rev: AIR_SCHEMA_REV,
            };
            let witness = vec![q(0.763); trace_length]; // Below threshold
            (public, witness)
        }
        "release" => {
            let public = PublicInputs {
                beta: q(0.85),
                w: 3,
                k: 2,
                m: 3,
                threshold: q(0.90),
                ema_init: q(0.946),
                viol_init: 0,
                clear_init: 2,
                quar_init: 1, // Already quarantined
                round_init: 9,
                quar_out: 0, // Release
                trace_length,
                // Provenance (prover will populate)
                prov_hash: [0, 0, 0, 0],
                profile_id: 128,
                air_rev: AIR_SCHEMA_REV,
            };
            let witness = vec![q(0.961); trace_length]; // Above threshold
            (public, witness)
        }
        _ => panic!("Unknown scenario: {}", scenario),
    }
}

fn mean(values: &[u64]) -> f64 {
    values.iter().sum::<u64>() as f64 / values.len() as f64
}

fn std_dev(values: &[u64], mean: f64) -> f64 {
    let variance = values
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn percentile(values: &mut [u64], p: f64) -> u64 {
    values.sort();
    let idx = ((values.len() as f64 * p) as usize).min(values.len() - 1);
    values[idx]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    println!("🔬 Winterfell PoGQ Benchmark");
    println!("============================");
    println!("Backend: Winterfell v0.13");
    println!("Scenario: {}", cli.scenario);
    println!("Trace Length: {} steps", cli.trace_length);
    println!("Rounds: {}\n", cli.rounds);

    let (public, witness) = get_scenario_inputs(&cli.scenario, cli.trace_length);
    let prover = PoGQProver::new();

    let mut prove_times = Vec::new();
    let mut verify_times = Vec::new();
    let mut proof_sizes = Vec::new();

    for round in 0..cli.rounds {
        print!("Round {}/{}: ", round + 1, cli.rounds);

        // Prove
        let prove_start = Instant::now();
        let result = prover.prove_exec(public.clone(), witness.clone())?;
        let prove_time = prove_start.elapsed().as_millis() as u64;
        prove_times.push(prove_time);
        proof_sizes.push(result.proof_bytes.len());

        print!("prove {}ms, ", prove_time);

        // Verify
        let verify_start = Instant::now();
        let valid = prover.verify_proof(&result.proof_bytes, public.clone())?;
        let verify_time = verify_start.elapsed().as_millis() as u64;
        verify_times.push(verify_time);

        println!("verify {}ms, size {}KB, valid={}",
                 verify_time, result.proof_bytes.len() / 1024, valid);

        if !valid {
            return Err("Verification failed!".into());
        }
    }

    // Compute statistics
    let mean_prove = mean(&prove_times);
    let std_prove = std_dev(&prove_times, mean_prove);
    let mut prove_sorted = prove_times.clone();
    let p95_prove = percentile(&mut prove_sorted, 0.95);

    let mean_verify = mean(&verify_times);
    let std_verify = std_dev(&verify_times, mean_verify);
    let mut verify_sorted = verify_times.clone();
    let p95_verify = percentile(&mut verify_sorted, 0.95);

    let mean_size = proof_sizes.iter().sum::<usize>() as f64 / proof_sizes.len() as f64;

    println!("\n📊 Summary Statistics");
    println!("=====================");
    println!("Prove time:  {:.1} ± {:.1} ms (p95: {} ms)", mean_prove, std_prove, p95_prove);
    println!("Verify time: {:.1} ± {:.1} ms (p95: {} ms)", mean_verify, std_verify, p95_verify);
    println!("Proof size:  {:.1} KB", mean_size / 1024.0);

    // Save results
    let result = BenchmarkResult {
        backend: "Winterfell".to_string(),
        scenario: cli.scenario.clone(),
        trace_length: cli.trace_length,
        rounds: cli.rounds,
        prove_times_ms: prove_times,
        verify_times_ms: verify_times,
        proof_sizes_bytes: proof_sizes,
        mean_prove_ms: mean_prove,
        std_prove_ms: std_prove,
        p95_prove_ms: p95_prove,
        mean_verify_ms: mean_verify,
        std_verify_ms: std_verify,
        p95_verify_ms: p95_verify,
        mean_proof_size_kb: mean_size / 1024.0,
    };

    // Create output directory if needed
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(&result)?;
    fs::write(&cli.output, json)?;
    println!("\n✅ Results saved to {}", cli.output.display());

    Ok(())
}
